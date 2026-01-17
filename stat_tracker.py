import os
import time
import json
import logging
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any, Optional

import requests
from dotenv import load_dotenv
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import boto3

VersionStatTracker = "1.1.3"
# ========= БАЗОВЫЕ ПУТИ =========

BASE_DIR = "/opt/stat_tracker"
DATA_DIR = os.path.join(BASE_DIR, "data")
LOG_DIR = os.path.join(BASE_DIR, "logs")
SEGMENTS_DIR = os.path.join(DATA_DIR, "segments")
USERS_LISTS_DIR = os.path.join(DATA_DIR, "users_lists")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(SEGMENTS_DIR, exist_ok=True)
os.makedirs(USERS_LISTS_DIR, exist_ok=True)

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
        self.base_url_v3 = "https://ads.vk.com/api/v3"
        self.headers = {"Authorization": f"Bearer {self.token}"}

    def fetch(self, endpoint: str, params: Optional[Dict[str, Any]] = None, api_version: str = "v2") -> Optional[Dict[str, Any]]:
        base = self.base_url if api_version == "v2" else self.base_url_v3
        url = f"{base}{endpoint}"
        max_attempts = 5
        backoff = 2  # seconds

        for attempt in range(1, max_attempts + 1):
            try:
                r = requests.get(url, headers=self.headers, params=params, timeout=30)

                # Client errors that won't be fixed by retrying
                if r.status_code == 403:
                    logging.warning(f"[{self.name}] 403 Forbidden on {url} - skipping")
                    return None
                
                if r.status_code == 404:
                    logging.warning(f"[{self.name}] 404 Not Found on {url} - skipping")
                    return None

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


    def get_paginated(self, endpoint: str, base_params: Optional[Dict[str, Any]] = None, api_version: str = "v2") -> List[Dict[str, Any]]:
        all_items: List[Dict[str, Any]] = []
        offset = 0
        params = dict(base_params or {})
        if "limit" not in params:
            params["limit"] = 200

        while True:
            params["offset"] = offset
            data = self.fetch(endpoint, params, api_version)
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

    def _fetch_banner_stats(self, acc: AccountInfo, banner_ids: List[int]) -> Dict[int, Dict[str, Any]]:
        """
        Запрашивает статистику для баннеров пачками по 200.
        Возвращает dict: banner_id -> {base: {...}, uniques: {...}, video: {...}}
        """
        stats_map: Dict[int, Dict[str, Any]] = {}
        
        # Разбиваем на пачки по 200
        for i in range(0, len(banner_ids), 200):
            batch = banner_ids[i:i+200]
            ids_str = ",".join(str(bid) for bid in batch)
            
            data = acc.fetch(
                "/statistics/banners/summary.json",
                {
                    "id": ids_str,
                    "metrics": "base,uniques,video"
                }
            )
            
            if not data or "items" not in data:
                continue
            
            for item in data.get("items", []):
                bid = item.get("id")
                if bid is not None:
                    stats_map[bid] = item.get("total", {})
        
        logging.info(f"[{acc.name}] Fetched stats for {len(stats_map)} banners")
        return stats_map

    def _fetch_faststat(self, acc: AccountInfo, banner_ids: List[int]) -> Dict[int, Dict[str, Any]]:
        """
        Запрашивает faststat для баннеров пачками по 200.
        Возвращает dict: banner_id -> {
            "clicks_minutely": "0,1,0,4,..." (60 значений),
            "shows_minutely": "36,28,23,..." (60 значений),
            "clicks_last_30_min": sum,
            "shows_last_30_min": sum
        }
        """
        faststat_map: Dict[int, Dict[str, Any]] = {}
        
        # Разбиваем на пачки по 200
        for i in range(0, len(banner_ids), 200):
            batch = banner_ids[i:i+200]
            ids_str = ",".join(str(bid) for bid in batch)
            
            data = acc.fetch(
                "/statistics/faststat/banners.json",
                {"id": ids_str},
                api_version="v3"
            )
            
            if not data or "banners" not in data:
                continue
            
            banners_data = data.get("banners", {})
            for bid_str, stats in banners_data.items():
                bid = int(bid_str)
                minutely = stats.get("minutely", {})
                clicks = minutely.get("clicks", [])
                shows = minutely.get("shows", [])
                
                faststat_map[bid] = {
                    # Поминутные данные как строка через запятую (для нейронки)
                    "clicks_minutely": ",".join(str(c) for c in clicks) if clicks else "0",
                    "shows_minutely": ",".join(str(s) for s in shows) if shows else "0",
                    # Суммы для быстрого анализа
                    "clicks_last_30_min": sum(clicks) if clicks else 0,
                    "shows_last_30_min": sum(shows) if shows else 0,
                }
        
        return faststat_map

    def collect_data_for_account(self, acc: AccountInfo) -> List[Dict[str, Any]]:
        snapshot_at = (datetime.utcnow() + timedelta(hours=4)).isoformat()
        records: List[Dict[str, Any]] = []

        # ===== КАМПАНИИ (COMPANY) =====
        campaigns = acc.get_paginated(
            "/ad_plans.json",
            {
                "limit": 200,
                "_status__ne": "deleted",
                "fields": "id,name,created,status,ad_groups,budget_limit_day,objective,updated,budget_limit,date_start,date_end,max_price,priced_goal",
            },
        )

        # map: company_id -> данные компании
        company_map: Dict[int, Dict[str, Any]] = {}
        
        for camp in campaigns:
            cid = camp.get("id")
            if cid:
                priced_goal = camp.get("priced_goal", {}) or {}
                company_map[cid] = {
                    "id_company": cid,
                    "name_company": camp.get("name"),
                    "created_company": camp.get("created"),
                    "updated_company": camp.get("updated"),
                    "budget_limit_day_company": camp.get("budget_limit_day"),
                    "budget_limit_company": camp.get("budget_limit"),
                    "date_start_company": camp.get("date_start"),
                    "date_end_company": camp.get("date_end"),
                    "max_price_company": camp.get("max_price"),
                    "status_company": camp.get("status"),
                    "objective_company": camp.get("objective"),
                    "priced_goal_name_company": priced_goal.get("name") or "0",
                    "priced_goal_source_id_company": priced_goal.get("source_id") or 0,
                }
        

        # ===== ГРУППЫ (GROUP) =====
        groups = acc.get_paginated(
            "/ad_groups.json",
            {
                "limit": 200,
                "_status__ne": "deleted",
                "fields": "name,created,updated,package_id,ad_plan_id,budget_limit_day,budget_limit,max_price,date_start,date_end,price,priced_goal,targetings,utm,objective",
            },
        )

        # map: group_id -> данные группы + компания
        group_info_map: Dict[int, Dict[str, Any]] = {}
        
        for g in groups:
            gid = g.get("id")
            if not gid:
                continue
            
            targetings = g.get("targetings", {}) or {}
            priced_goal_group = g.get("priced_goal", {}) or {}
            
            # === FULLTIME ===
            fulltime = targetings.get("fulltime", {}) or {}
            fulltime_flags = ",".join(fulltime.get("flags", [])) if fulltime.get("flags") else "0"
            
            # Дни недели
            days = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
            fulltime_schedule = {}
            for day in days:
                hours = fulltime.get(day, [])
                fulltime_schedule[f"fulltime_{day}"] = self._join_or_zero(hours) if hours else "0"
        
            group_info_map[gid] = {
                "id_group": gid,
                "name_group": g.get("name"),
                "created_group": g.get("created"),
                "updated_group": g.get("updated"),
                "status_group": g.get("status"),
                "package_id_group": g.get("package_id"),
                "budget_limit_day_group": g.get("budget_limit_day"),
                "budget_limit_group": g.get("budget_limit"),
                "max_price_group": g.get("max_price"),
                "date_start_group": g.get("date_start"),
                "date_end_group": g.get("date_end"),
                "price_group": g.get("price"),
                "utm_group": g.get("utm") or "0",
                "objective_group": g.get("objective") or "0",
                "priced_goal_name_group": priced_goal_group.get("name") or "0",
                "priced_goal_source_id_group": priced_goal_group.get("source_id") or 0,
                "age": self._parse_age(targetings),
                "geo": self._join_or_zero(targetings.get("geo", {}).get("regions")),
                "pads": self._join_or_zero(targetings.get("pads")),
                "group_members": targetings.get("group_members") or "0",
                "interests": self._join_or_zero(targetings.get("interests")),
                "id_segments": self._join_or_zero(targetings.get("segments")),
                "sex": self._join_or_zero(targetings.get("sex")),
                "fulltime_flags": fulltime_flags,
                **fulltime_schedule,
            }
        
            # добавляем данные компании по ad_plan_id
            company_id = g.get("ad_plan_id")
            if company_id:
                group_info_map[gid].update(company_map.get(company_id, {}))

        # ===== БАННЕРЫ =====
        banners = acc.get_paginated(
            "/banners.json",
            {
                "limit": 200,
                "_status__ne": "deleted",
                "fields": "id,name,created,updated,status,ad_group_id,content,textblocks,urls",
            },
        )

        # ===== СТАТИСТИКА ПО БАННЕРАМ (с пагинацией по id) =====
        banner_ids = [ban.get("id") for ban in banners if ban.get("id") is not None]
        stats_map = self._fetch_banner_stats(acc, banner_ids)

        # ===== FASTSTAT (поминутная статистика за 30 мин) =====
        faststat_map = self._fetch_faststat(acc, banner_ids)

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
                "created_banner": ban.get("created"),
                "updated_banner": ban.get("updated"),
            }

            # ===== TEXTBLOCKS =====
            tb = ban.get("textblocks", {}) or {}
            
            # === TITLE (title_40_vkads) ===
            title = tb.get("title_40_vkads", {}).get("text") or "0"
            
            # === ABOUT COMPANY ===
            about_company = tb.get("about_company_115", {}).get("text") or "0"
            
            # === CTA (cta_leadads, cta_community_vk, cta_sites_full и т.д.) ===
            cta = "0"
            for key in tb.keys():
                if key.startswith("cta_"):
                    cta = tb.get(key, {}).get("text") or "0"
                    break
            
            # === SHORT TEXT (короткий: text_90, text_50) ===
            text_short = (
                tb.get("text_90", {}).get("text")
                or tb.get("text_50", {}).get("text")
                or "0"
            )
            
            # === LONG TEXT (длинный: text_long, text_2000, text_220) ===
            text_long = (
                tb.get("text_long", {}).get("text")
                or tb.get("text_2000", {}).get("text")
                or tb.get("text_220", {}).get("text")
            )
            
            # === Если не нашли long — ищем любой text_* ===
            if not text_long or text_long == "":
                for k, v in tb.items():
                    if k.startswith("text_") and isinstance(v, dict) and "text" in v:
                        if v.get("text"):
                            text_long = v.get("text")
                            break
                        
            # Если снова ничего — ставим "0"
            text_long = text_long or "0"

            
            # ===== ICON (icon_256x256) =====
            content = ban.get("content", {}) or {}
            icon = content.get("icon_256x256", {}) or {}
            icon_id = icon.get("id") or 0
            icon_url = icon.get("variants", {}).get("original", {}).get("url") or "0"
            
            # ===== IMAGES =====
            
            # список всех найденных id и URL
            image_ids = []
            image_urls = []
            
            # приоритетная картинка
            image600 = content.get("image_600x600")
            if image600:
                variants = image600.get("variants", {}) or {}
                url = variants.get("original", {}).get("url")
                if url:
                    image_ids.append(str(image600.get("id", "")))
                    image_urls.append(url)
            
            # если image_600x600 нет — ищем ВСЕ image_*
            if not image_ids:
                for key, img in content.items():
                    if not key.startswith("image_"):
                        continue
                    
                    if not isinstance(img, dict):
                        continue
                    
                    img_id = img.get("id")
                    variants = img.get("variants", {}) or {}
                    url = variants.get("original", {}).get("url")
            
                    if img_id and url:
                        image_ids.append(str(img_id))
                        image_urls.append(url)
            
            # формируем значения
            id_image = ",".join(image_ids) if image_ids else "0"
            image_url = ",".join(image_urls) if image_urls else "0"

            
            # ===== VIDEO =====
            video_variants = []  # названия вариантов видео (video_portrait_9_16_180s и т.д.)
            video_ids = []
            video_high_urls = []  # url из high
            video_preview_urls = []  # url из high-first_frame
            
            # Собираем ВСЕ ключи video_*
            for key, video in content.items():
                if not key.startswith("video_"):
                    continue
                
                if not isinstance(video, dict):
                    continue
                
                vid = video.get("id")
                variants = video.get("variants", {}) or {}
                
                # URL из high
                high_url = variants.get("high", {}).get("url") or ""
                # URL превью из high-first_frame
                preview_url = variants.get("high-first_frame", {}).get("url") or ""
                
                # Если нет high → пробуем internal или любой другой
                if not high_url:
                    high_url = variants.get("internal", {}).get("url") or ""
                if not high_url:
                    for v in variants.values():
                        if isinstance(v, dict) and v.get("url") and v.get("media_type") == "video":
                            high_url = v["url"]
                            break
                
                video_variants.append(key)
                if vid:
                    video_ids.append(str(vid))
                if high_url:
                    video_high_urls.append(high_url)
                if preview_url:
                    video_preview_urls.append(preview_url)
            
            # Собираем результат
            video_variants_str = ",".join(video_variants) if video_variants else "0"
            id_video = ",".join(video_ids) if video_ids else "0"
            video_url = ",".join(video_high_urls) if video_high_urls else "0"
            video_preview_url = ",".join(video_preview_urls) if video_preview_urls else "0"
            
            
            # ===== URLS =====
            urls = ban.get("urls", {}) or {}
            primary_url = urls.get("primary", {}) or {}
            banner_url = primary_url.get("url") or "0"
            banner_url_object_id = primary_url.get("url_object_id") or "0"
            banner_url_object_type = primary_url.get("url_object_type") or "0"
            
            
            # ===== ДАННЫЕ ГРУППЫ/КАМПАНИИ =====
            group_id = ban.get("ad_group_id")
            group_info = group_info_map.get(group_id, {})
            base_record.update(group_info)


            # статистика
            s = stats_map.get(ban_id, {})
            base = s.get("base", {}) or {}
            video_stats = s.get("video", {}) or {}
            vk_stats = base.get("vk", {}) or {}
            uniques_stats = s.get("uniques", {}) or {}
            
            # faststat
            fs = faststat_map.get(ban_id, {})

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
                    # uniques
                    "uniques_total": uniques_stats.get("total", 0),
                    "uniques_frequency": float(uniques_stats.get("frequency", 0) or 0),
                    # faststat - поминутные данные (60 значений через запятую)
                    "clicks_minutely": fs.get("clicks_minutely", "0"),
                    "shows_minutely": fs.get("shows_minutely", "0"),
                    # faststat - суммы за последние 30 мин
                    "clicks_last_30_min": fs.get("clicks_last_30_min", 0),
                    "shows_last_30_min": fs.get("shows_last_30_min", 0),
                    # textblocks
                    "title": title or "0",
                    "text_short": text_short or "0",
                    "text_long": text_long or "0",
                    "about_company": about_company or "0",
                    "cta": cta or "0",
                    # icon
                    "icon_id": icon_id or 0,
                    "icon_url": icon_url or "0",
                    # video
                    "video_variants": video_variants_str or "0",
                    "id_video": id_video or "0",
                    "video_url": video_url or "0",
                    "video_preview_url": video_preview_url or "0",
                    # image
                    "id_image": id_image or "0",
                    "image_url": image_url or "0",
                    # urls
                    "banner_url": banner_url or "0",
                    "banner_url_object_id": banner_url_object_id or "0",
                    "banner_url_object_type": banner_url_object_type or "0",
                    # video stats
                    "viewed_25_percent_rate": float(video_stats.get("viewed_25_percent_rate", 0) or 0),
                    "viewed_50_percent_rate": float(video_stats.get("viewed_50_percent_rate", 0) or 0),
                    "viewed_75_percent_rate": float(video_stats.get("viewed_75_percent_rate", 0) or 0),
                    "viewed_100_percent_rate": float(video_stats.get("viewed_100_percent_rate", 0) or 0),
                }
            )

            records.append(base_record)

        logging.info(f"[{acc.name}] Collected {len(records)} records")
        return records

    def collect_and_save_per_account(self):
        """
        Собирает данные и сохраняет parquet по каждому аккаунту отдельно.
        Это экономит RAM.
        """
        now = datetime.utcnow() + timedelta(hours=4)
        day_folder = now.strftime("%d_%m_%Y")
        timestamp = now.strftime("%d_%m_%Y_%H-%M-%S")
        
        for acc in self.accounts:
            try:
                records = self.collect_data_for_account(acc)
                
                if not records:
                    logging.warning(f"[{acc.name}] No records collected, skipping.")
                    continue
                
                df = pd.DataFrame(records)
                
                # Локальный файл
                local_path = os.path.join(DATA_DIR, f"stat_{acc.name}.parquet")
                table = pa.Table.from_pandas(df)
                
                # Удаляем старый если есть
                if os.path.exists(local_path):
                    os.remove(local_path)
                
                pq.write_table(table, local_path)
                logging.info(f"[{acc.name}] Saved parquet: {local_path}")
                
                # Заливаем в S3
                s3_key = f"stat_tracker_parquet/{day_folder}/{acc.name}/stat_{acc.name}_{timestamp}.parquet"
                try:
                    s3_client.upload_file(local_path, S3_BUCKET, s3_key)
                    logging.info(f"[{acc.name}] Uploaded to S3: s3://{S3_BUCKET}/{s3_key}")
                except Exception as e:
                    logging.error(f"[{acc.name}] Failed to upload to S3: {e}")
                
                # Очищаем память
                del df
                del records
                
            except Exception as e:
                logging.error(f"Error processing account {acc.name}: {e}")

    def collect_data(self) -> pd.DataFrame:
        """Старый метод для совместимости - собирает все в один DataFrame"""
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

        now = datetime.utcnow() + timedelta(hours=4)
        day_folder = now.strftime("%d_%m_%Y")
        timestamp = (datetime.utcnow() + timedelta(hours=4)).strftime("%d_%m_%Y_%H-%M-%S")
        key = f"stat_tracker_parquet/{day_folder}/stat_tracker_{timestamp}.parquet"

        # Заливаем новый
        try:
            s3_client.upload_file(PARQUET_PATH, S3_BUCKET, key)
            logging.info(f"Uploaded parquet to S3: s3://{S3_BUCKET}/{key}")
        except Exception as e:
            logging.error(f"Failed to upload parquet to S3: {e}")


class AudienceTracker:
    """
    Класс для сбора аудиторий (segments), их relations и users_lists.
    """
    
    def __init__(self, accounts: List[AccountInfo]):
        self.accounts = accounts
    
    def _get_offset_file(self, acc_name: str) -> str:
        return os.path.join(SEGMENTS_DIR, f"{acc_name}_offset.json")
    
    def _load_offset(self, acc_name: str) -> Dict[str, int]:
        """Загружает offset и last_total_count"""
        offset_file = self._get_offset_file(acc_name)
        if os.path.exists(offset_file):
            try:
                with open(offset_file, "r") as f:
                    data = json.load(f)
                    return {
                        "offset": data.get("offset", 0),
                        "last_total_count": data.get("last_total_count", 0)
                    }
            except:
                pass
        return {"offset": 0, "last_total_count": 0}
    
    def _save_offset(self, acc_name: str, offset: int, total_count: int):
        offset_file = self._get_offset_file(acc_name)
        with open(offset_file, "w") as f:
            json.dump({"offset": offset, "last_total_count": total_count}, f)
    
    def collect_segments_for_account(self, acc: AccountInfo):
        """
        Собирает segments для аккаунта с пагинацией.
        Один запуск = одна страница (100 записей).
        """
        offset_data = self._load_offset(acc.name)
        offset = offset_data["offset"]
        last_total_count = offset_data["last_total_count"]
        
        data = acc.fetch(
            "/remarketing/segments.json",
            {
                "limit": 100,
                "offset": offset,
                "fields": "id,name,created,relations,pass_condition"
            }
        )
        
        if not data or "items" not in data:
            logging.warning(f"[{acc.name}] No segments data received")
            return
        
        items = data.get("items", [])
        total_count = data.get("count", 0)
        
        # Если offset >= total_count и total не изменился — скип
        if offset >= total_count and total_count == last_total_count:
            logging.info(f"[{acc.name}] Segments already synced (offset: {offset}, total: {total_count}), skipping")
            return
        
        # Если offset >= total_count но total изменился — сбрасываем и начинаем сначала
        if offset >= total_count and total_count != last_total_count:
            logging.info(f"[{acc.name}] Total count changed ({last_total_count} -> {total_count}), resetting offset")
            offset = 0
            self._save_offset(acc.name, 0, total_count)
            # Делаем новый запрос с offset=0
            data = acc.fetch(
                "/remarketing/segments.json",
                {
                    "limit": 100,
                    "offset": 0,
                    "fields": "id,name,created,relations,pass_condition"
                }
            )
            if not data or "items" not in data:
                return
            items = data.get("items", [])
            total_count = data.get("count", 0)
        
        if not items:
            logging.info(f"[{acc.name}] No segments found")
            self._save_offset(acc.name, 0, total_count)
            return
            self._save_offset(acc.name, 0)
            return
        
        # Собираем записи для parquet
        records = []
        for seg in items:
            seg_id = seg.get("id")
            relations = seg.get("relations", []) or []
            
            # Собираем object_type из relations
            object_types = []
            for rel in relations:
                ot = rel.get("object_type")
                if ot:
                    object_types.append(ot)
            
            # Запрашиваем детальные relations для получения source_id и type
            relations_details = self._fetch_segment_relations(acc, seg_id)
            
            # Собираем source_id и type из relations
            source_ids = []
            relation_types = []
            for rd in relations_details:
                if rd.get("source_id"):
                    source_ids.append(str(rd["source_id"]))
                if rd.get("type"):
                    relation_types.append(rd["type"])
            
            records.append({
                "account_name": acc.name,
                "id": seg_id,
                "name": seg.get("name"),
                "created": seg.get("created"),
                "relations_object_type": ",".join(object_types) if object_types else "0",
                "pass_condition": seg.get("pass_condition"),
                "relations_source_id": ",".join(source_ids) if source_ids else "0",
                "relations_type": ",".join(relation_types) if relation_types else "0",
            })
        
        # Дозаписываем в parquet
        if records:
            self._append_to_segments_parquet(acc.name, records)
        
        # Обновляем offset
        new_offset = offset + len(items)
        self._save_offset(acc.name, new_offset, total_count)
        
        logging.info(f"[{acc.name}] Collected {len(items)} segments (offset: {offset} -> {new_offset}, total: {total_count})")
    
    def _fetch_segment_relations(self, acc: AccountInfo, segment_id: int) -> List[Dict[str, Any]]:
        """
        Запрашивает relations для конкретного сегмента.
        Возвращает список с source_id и type из params.
        """
        data = acc.fetch(
            f"/remarketing/segments/{segment_id}/relations.json",
            {"fields": "id,object_id,object_type,params"}
        )
        
        if not data or "items" not in data:
            return []
        
        items = data.get("items", [])
        
        # Собираем данные relations
        relations_data = []
        for rel in items:
            params = rel.get("params", {}) or {}
            relations_data.append({
                "source_id": params.get("source_id"),
                "type": params.get("type"),
            })
        
        return relations_data
    
    def _append_to_segments_parquet(self, acc_name: str, records: List[Dict[str, Any]]):
        """
        Дозаписывает записи в parquet файл аудиторий.
        """
        parquet_path = os.path.join(SEGMENTS_DIR, f"{acc_name}_segments.parquet")
        
        new_df = pd.DataFrame(records)
        
        if os.path.exists(parquet_path):
            # Читаем существующий и добавляем
            existing_df = pd.read_parquet(parquet_path)
            
            # Удаляем дубликаты по id (оставляем новые)
            existing_ids = set(existing_df["id"].tolist())
            new_records = [r for r in records if r["id"] not in existing_ids]
            
            if new_records:
                new_df = pd.DataFrame(new_records)
                combined_df = pd.concat([existing_df, new_df], ignore_index=True)
                table = pa.Table.from_pandas(combined_df)
                pq.write_table(table, parquet_path)
                logging.info(f"[{acc_name}] Appended {len(new_records)} new segments to parquet")
            else:
                logging.info(f"[{acc_name}] No new segments to append")
        else:
            # Создаем новый
            table = pa.Table.from_pandas(new_df)
            pq.write_table(table, parquet_path)
            logging.info(f"[{acc_name}] Created segments parquet with {len(records)} records")
    
    def collect_users_lists_for_account(self, acc: AccountInfo):
        """
        Собирает users_lists для аккаунта и сохраняет в JSON.
        Запрос без лимита - получаем все сразу.
        """
        data = acc.fetch(
            "/remarketing/users_lists.json",
            {"fields": "id,name,created,entries_count,status,type"},
            api_version="v3"
        )
        
        if not data or "items" not in data:
            logging.warning(f"[{acc.name}] No users_lists data received")
            return
        
        items = data.get("items", [])
        
        # Сохраняем в JSON для быстрого поиска
        json_path = os.path.join(USERS_LISTS_DIR, f"{acc.name}_users_lists.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=False, indent=2)
        
        logging.info(f"[{acc.name}] Saved {len(items)} users_lists to {json_path}")
        
        # Также заливаем в S3
        now = datetime.utcnow() + timedelta(hours=4)
        day_folder = now.strftime("%d_%m_%Y")
        s3_key = f"audiences/{day_folder}/{acc.name}_users_lists.json"
        
        try:
            s3_client.upload_file(json_path, S3_BUCKET, s3_key)
            logging.info(f"[{acc.name}] Uploaded users_lists to S3: s3://{S3_BUCKET}/{s3_key}")
        except Exception as e:
            logging.error(f"[{acc.name}] Failed to upload users_lists to S3: {e}")
    
    def collect_all(self):
        """
        Собирает данные аудиторий для всех аккаунтов.
        """
        for acc in self.accounts:
            try:
                logging.info(f"[{acc.name}] Starting audience collection...")
                
                # Сегменты (одна страница за запуск)
                self.collect_segments_for_account(acc)
                
                # Users lists (все сразу)
                self.collect_users_lists_for_account(acc)
                
            except Exception as e:
                logging.error(f"[{acc.name}] Error collecting audiences: {e}")
        
        # Заливаем parquet сегментов в S3
        self._upload_segments_to_s3()
    
    def _upload_segments_to_s3(self):
        """
        Заливает все parquet файлы сегментов в S3.
        """
        now = datetime.utcnow() + timedelta(hours=4)
        day_folder = now.strftime("%d_%m_%Y")
        
        for filename in os.listdir(SEGMENTS_DIR):
            if filename.endswith("_segments.parquet"):
                local_path = os.path.join(SEGMENTS_DIR, filename)
                s3_key = f"audiences/{day_folder}/{filename}"
                
                try:
                    s3_client.upload_file(local_path, S3_BUCKET, s3_key)
                    logging.info(f"Uploaded segments to S3: s3://{S3_BUCKET}/{s3_key}")
                except Exception as e:
                    logging.error(f"Failed to upload {filename} to S3: {e}")


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
        AccountInfo(
            token=os.getenv("VK_TOKEN_ZEL_1"),
            name="ZEL_1",
        ),
        AccountInfo(
            token=os.getenv("VK_TOKEN_ZEL_2"),
            name="ZEL_2",
        ),
        AccountInfo(
            token=os.getenv("VK_TOKEN_ROM_1"),
            name="ROM_1",
        ),
        AccountInfo(
            token=os.getenv("VK_TOKEN_GUZ_1"),
            name="GUZ_1",
        ),
        AccountInfo(
            token=os.getenv("VK_TOKEN_GUZ_2"),
            name="GUZ_2",
        ),
        AccountInfo(
            token=os.getenv("VK_TOKEN_GUZ_3"),
            name="GUZ_3",
        ),
        AccountInfo(
            token=os.getenv("VK_TOKEN_GUZ_4"),
            name="GUZ_4",
        ),
        AccountInfo(
            token=os.getenv("VK_TOKEN_GUZ_5"),
            name="GUZ_5",
        ),
        AccountInfo(
            token=os.getenv("VK_TOKEN_ALE_1"),
            name="ALE_1",
        ),
        AccountInfo(
            token=os.getenv("VK_TOKEN_NIKOLAY_1"),
            name="NIKOLAY_1",
        ),
        AccountInfo(
            token=os.getenv("VK_TOKEN_NIKOLAY_2"),
            name="NIKOLAY_2",
        ),
        AccountInfo(
            token=os.getenv("VK_TOKEN_NIKOLAY_3"),
            name="NIKOLAY_3",
        ),
        AccountInfo(
            token=os.getenv("VK_TOKEN_NIKOLAY_4"),
            name="NIKOLAY_4",
        ),
        AccountInfo(
            token=os.getenv("VK_TOKEN_NIKOLAY_5"),
            name="NIKOLAY_5",
        ),
        AccountInfo(
            token=os.getenv("VK_TOKEN_NIKOLAY_6"),
            name="NIKOLAY_6",
        ),
        AccountInfo(
            token=os.getenv("VK_TOKEN_NIKOLAY_7"),
            name="NIKOLAY_7",
        ),
        AccountInfo(
            token=os.getenv("VK_TOKEN_OLYA_1"),
            name="OLYA_1",
        ),
        AccountInfo(
            token=os.getenv("VK_TOKEN_OLYA_2"),
            name="OLYA_2",
        ),
    ]

    # ===== ОСНОВНОЙ СБОР СТАТИСТИКИ (по аккаунтам) =====
    tracker = StatTracker(ACCOUNTS)
    tracker.collect_and_save_per_account()
    logging.info("Stat tracking completed.")
    
    # ===== СБОР АУДИТОРИЙ =====
    audience_tracker = AudienceTracker(ACCOUNTS)
    audience_tracker.collect_all()
    logging.info("Audience tracking completed.")
