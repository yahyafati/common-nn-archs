from datetime import datetime


def generate_run_id(fmt: str = "CNA_%Y%m%d%H%M%S%z") -> str:
    """
    Generate a run ID using strftime format codes.

    Common codes:
        %Y  Year (2026)     %m  Month (01-12)    %d  Day (01-31)
        %H  Hour 24 (00-23) %M  Minute (00-59)   %S  Second (00-59)
        %f  Microsecond (000000-999999)
        %z  Timezone (+0800)  %Z  Timezone name (UTC, CST)

    Docs: https://docs.python.org/3/library/datetime.html#strftime-and-strptime-format-codes
    """
    return datetime.now().astimezone().strftime(fmt)
