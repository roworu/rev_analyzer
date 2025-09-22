import logging
import useful.logs

STATIC_FIELDS = {
    "application": "rev_analyzer",
}

JSON_FIELDS = {
    "@timestamp": "asctime",
    "message": "message",
    "time": "created",
    "log_level": "levelname"
}

useful.logs.setup(json_fields=JSON_FIELDS, always_extra=STATIC_FIELDS)
_log = logging.getLogger(__name__)

if __name__ == "__main__":
    _log.info("Starting rev_analyzer application")
    from app.app import start_app
    start_app()
    
