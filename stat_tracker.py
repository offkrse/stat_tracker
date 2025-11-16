import os
import time
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional

import requests
from dotenv import load_dotenv
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import boto3

VersionStatTracker = "0.3"
# ========= БАЗОВЫЕ ПУТИ =========

BASE_DIR = "/opt/stat_tracker"
DATA_DIR = os.path.join(BASE_DIR, "data")
LOG_DIR = os.path.join(BASE_DIR, "logs")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

PARQUET_PATH = os.path.join(DATA_DIR, "stat_data.parquet")

# ========= ЛОГИ =========

logging.basicConfig(
    filename=os.path.join(LOG_DIR, "stat_tracker.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

console = logging.StreamHandler()
console.setLevel(logging.INFO)
console.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logging.getLogger("").addHandler(console)

# ========= ENV & S3 =========

load_dotenv()

S3_ENDPOINT = os.getenv("S3_ENDPOINT")
S3_BUCKET = os.getenv("S3_BUCKET")
S3_ACCESS_KEY = os.getenv("S3_ACCESS_KEY")
S3_SECRET_KEY = os.getenv("S3_SECRET_KEY")

s3_client = boto3.client(
    "s3",
    endpoint_url=S3_ENDPOINT,
    aws_access_key_id=S3_ACCESS_KEY,
    aws_secret_access_key=S3_SECRET_KEY,
)


class AccountInfo:

    def __init__(self, token: str, name: str = ""):
        self.token = token
        self.name = name
        self.base_url = "https://ads.vk.com/api/v2"
        self.headers = {"Authorization": f"Bearer {self.token}"}

    def fetch(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        url = f"{self.base_url}{endpoint}"
        max_attempts = 5
        backoff = 2  # seconds

        for attempt in range(1, max_attempts + 1):
            try:
                r = requests.get(url, headers=self.headers, params=params, timeout=30)

                # Flood control: VK sends HTTP 429
                if r.status_code == 429:
                    retry_after = int(r.headers.get("Retry-After", backoff))
                    logging.warning(
                        f"[{self.name}] 429 Flood limit on {url}, retrying in {retry_after}s (attempt {attempt}/{max_attempts})"
                    )
                    time.sleep(retry_after)
                    continue

                # Server errors
                if r.status_code >= 500:
                    logging.warning(
                        f"[{self.name}] Server error {r.status_code} on {url}, retrying in {backoff}s (attempt {attempt}/{max_attempts})"
                    )
                    time.sleep(backoff)
                    backoff *= 2
                    continue

                r.raise_for_status()
                return r.json()

            except requests.exceptions.RequestException as e:
                logging.error(
                    f"[{self.name}] Network/API error on {url}: {e} (attempt {attempt}/{max_attempts})"
                )
                time.sleep(backoff)
                backoff *= 2

        logging.error(f"[{self.name}] FAILED all retries for {url}")
        return None


    def get_paginated(self, endpoint: str, base_params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        all_items: List[Dict[str, Any]] = []
        offset = 0
        params = dict(base_params or {})
        if "limit" not in params:
            params["limit"] = 200

        while True:
            params["offset"] = offset
            data = self.fetch(endpoint, params)
            if not data or "items" not in data:
                break

            items = data["items"]
            all_items.extend(items)

            if len(items) < params.get("limit", 200):
                break

            offset += params["limit"]

        logging.info(f"[{self.name}] Fetched {len(all_items)} items from {endpoint}")
        return all_items


class StatTracker:
    def __init__(self, accounts: List[AccountInfo]):
        self.accounts = accounts

    def _parse_age(self, targetings: Dict[str, Any]) -> str:
        """
        age_list -> строка формата "21-50".
        Если нет списка или только 0 — возвращаем "0".
        """
        age_info = targetings.get("age", {})
        age_list = age_info.get("age_list") or []
        ages = [a for a in age_list if a != 0]
        if not ages:
            return "0"
        return f"{min(ages)}-{max(ages)}"

    def _join_or_zero(self, values: Optional[List[Any]]) -> str:
        if not values:
            return "0"
        return ",".join(str(v) for v in values)

    def collect_data_for_account(self, acc: AccountInfo) -> List[Dict[str, Any]]:
        snapshot_at = datetime.utcnow().isoformat()
        records: List[Dict[str, Any]] = []

        # ===== КАМПАНИИ (COMPANY) =====
        campaigns = acc.get_paginated(
            "/ad_plans.json",
            {
                "limit": 200,
                "_status__ne": "deleted",
                "fields": "id,name,created,status,ad_groups,budget_limit_day,objective",
            },
        )

        # map: group_id -> данные компании
        group_to_company: Dict[int, Dict[str, Any]] = {}
        for camp in campaigns:
            company_id = camp.get("id")
            ad_groups = camp.get("ad_groups") or []
            for g in ad_groups:
                gid = g.get("id")
                if gid:
                    group_to_company[gid] = {
                        "id_company": company_id,
                        "name_company": camp.get("name"),
                        "created_company": camp.get("created"),
                        "budget_limit_day_company": camp.get("budget_limit_day"),
                        "status_company": camp.get("status"),
                        "objective_company": camp.get("objective"),
                    }

        # ===== ГРУППЫ (GROUP) =====
        groups = acc.get_paginated(
            "/ad_groups.json",
            {
                "limit": 200,
                "_status__ne": "deleted",
                "fields": "id,name,created,status,banners,budget_limit_day,targetings",
            },
        )

        # map: banner_id -> данные группы + таргетинги
        banner_to_group: Dict[int, Dict[str, Any]] = {}
        for g in groups:
            gid = g.get("id")
            targetings = g.get("targetings", {})

            age_str = self._parse_age(targetings)
            geo_str = self._join_or_zero(targetings.get("geo", {}).get("regions"))
            interests_str = self._join_or_zero(targetings.get("interests"))
            segments_str = self._join_or_zero(targetings.get("segments"))
            sex_values = targetings.get("sex") or []
            sex_str = self._join_or_zero(sex_values)

            group_base = {
                "id_group": gid,
                "name_group": g.get("name"),
                "created_group": g.get("created"),
                "status_group": g.get("status"),
                "budget_limit_day_group": g.get("budget_limit_day"),
                "age": age_str,
                "geo": geo_str,
                "interests": interests_str,
                "id_segments": segments_str,
                "sex": sex_str,
            }

            # прикрепляем компанию, если есть
            company_info = group_to_company.get(gid, {})
            group_base.update(company_info)

            banners = g.get("banners") or []
            for b in banners:
                bid = b.get("id")
                if not bid:
                    continue
                banner_to_group[bid] = group_base

        # ===== БАННЕРЫ =====
        banners = acc.get_paginated(
            "/banners.json",
            {
                "limit": 200,
                "_status__ne": "deleted",
                "fields": "id,name,created,status,content,moderation_status,textblocks",
            },
        )

        # ===== СТАТИСТИКА ПО БАННЕРАМ =====
        stats = acc.fetch(
            "/statistics/banners/summary.json",
            {"metrics": "base,video"},
        ) or {}

        stats_map: Dict[int, Dict[str, Any]] = {}
        for item in stats.get("items", []):
            bid = item.get("id")
            if bid is not None:
                stats_map[bid] = item.get("total", {})

        # ===== СБОР ЗАПИСЕЙ =====
        for ban in banners:
            ban_id = ban.get("id")
            if ban_id is None:
                continue

            base_record: Dict[str, Any] = {
                "snapshot_at": snapshot_at,
                "account_name": acc.name,
                "id_banner": ban_id,
                "name_banner": ban.get("name"),
                "status_banner": ban.get("status"),
                "moderation_status_banner": ban.get("moderation_status"),
                "created_banner": ban.get("created"),
            }

            # текстовые поля
            tb = ban.get("textblocks", {}) or {}
            base_record.update(
                {
                    "title": tb.get("title_40_vkads", {}).get("text", "0"),
                    "text_long": tb.get("text_220", {}).get("text", "0"),
                    "text_short": tb.get("text_90", {}).get("text", "0"),
                }
            )

            # video id
            video = (ban.get("content") or {}).get("video_portrait_9_16_180s", {}) or {}
            base_record["id_video"] = video.get("id", 0)

            # данные группы/кампании
            group_info = banner_to_group.get(ban_id, {})
            base_record.update(group_info)

            # статистика
            s = stats_map.get(ban_id, {})
            base = s.get("base", {}) or {}
            video_stats = s.get("video", {}) or {}
            vk_stats = base.get("vk", {}) or {}

            base_record.update(
                {
                    "shows": base.get("shows", 0),
                    "clicks": base.get("clicks", 0),
                    "spent": float(base.get("spent", 0) or 0),
                    "cpm": float(base.get("cpm", 0) or 0),
                    "cpc": float(base.get("cpc", 0) or 0),
                    "ctr": float(base.get("ctr", 0) or 0),
                    "goals": vk_stats.get("goals", 0),
                    "cpa": float(vk_stats.get("cpa", 0) or 0),
                    "cr": float(vk_stats.get("cr", 0) or 0),
                    "viewed_25_percent_rate": float(video_stats.get("viewed_25_percent_rate", 0) or 0),
                    "viewed_50_percent_rate": float(video_stats.get("viewed_50_percent_rate", 0) or 0),
                    "viewed_75_percent_rate": float(video_stats.get("viewed_75_percent_rate", 0) or 0),
                    "viewed_100_percent_rate": float(video_stats.get("viewed_100_percent_rate", 0) or 0),
                }
            )

            records.append(base_record)

        logging.info(f"[{acc.name}] Collected {len(records)} records")
        return records

    def collect_data(self) -> pd.DataFrame:
        all_records: List[Dict[str, Any]] = []
        for acc in self.accounts:
            try:
                recs = self.collect_data_for_account(acc)
                all_records.extend(recs)
            except Exception as e:
                logging.error(f"Error processing account {acc.name}: {e}")

        if not all_records:
            logging.warning("No records collected from any account.")
            return pd.DataFrame()

        df = pd.DataFrame(all_records)
        logging.info(f"Total records across all accounts: {len(df)}")
        return df

    def save_parquet(self, df: pd.DataFrame):
        if df.empty:
            logging.warning("DataFrame is empty, parquet will not be written.")
            return

        table = pa.Table.from_pandas(df)

        if os.path.exists(PARQUET_PATH):
            try:
                os.remove(PARQUET_PATH)
                logging.info(f"Removed previous parquet: {PARQUET_PATH}")
            except Exception as e:
                logging.error(f"Failed to remove old parquet: {e}")

        pq.write_table(table, PARQUET_PATH)
        logging.info(f"New parquet file saved at {PARQUET_PATH}")

    def upload_to_s3(self):
        if not os.path.exists(PARQUET_PATH):
            logging.warning("Parquet file does not exist, nothing to upload.")
            return

        key_prefix = "stat_tracker_parquet/"
        key = key_prefix + "stat_tracker.parquet"

        # Удаляем предыдущие parquet в этой папке
        try:
            resp = s3_client.list_objects_v2(Bucket=S3_BUCKET, Prefix=key_prefix)
            for obj in resp.get("Contents", []):
                s3_client.delete_object(Bucket=S3_BUCKET, Key=obj["Key"])
                logging.info(f"Deleted old parquet from S3: {obj['Key']}")
        except Exception as e:
            logging.error(f"Failed to list/delete old parquet in S3: {e}")

        # Заливаем новый
        try:
            s3_client.upload_file(PARQUET_PATH, S3_BUCKET, key)
            logging.info(f"Uploaded parquet to S3: s3://{S3_BUCKET}/{key}")
        except Exception as e:
            logging.error(f"Failed to upload parquet to S3: {e}")


def load_accounts_from_env() -> List[AccountInfo]:
    accounts: List[AccountInfo] = []
    i = 1
    while True:
        token = os.getenv(f"VK_TOKEN_{i}")
        if not token:
            break
        name = os.getenv(f"VK_ACCOUNT_{i}_NAME", f"account_{i}")
        accounts.append(AccountInfo(token=token, name=name))
        i += 1

    if not accounts:
        logging.error("No VK_TOKEN_* found in .env")
    else:
        logging.info(f"Loaded {len(accounts)} accounts from env.")
    return accounts


if __name__ == "__main__":
    ACCOUNTS = [
        AccountInfo(
            token=os.getenv("VK_TOKEN_MAIN"),
            name="MAIN",
        ),
    ]


    tracker = StatTracker(ACCOUNTS)
    df = tracker.collect_data()
    tracker.save_parquet(df)
    tracker.upload_to_s3()
    logging.info("Stat tracking completed.")
