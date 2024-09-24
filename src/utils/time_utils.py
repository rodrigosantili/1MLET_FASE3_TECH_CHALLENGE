import datetime
import time


def convert_year_to_epoch(year: int) -> int:
    """
    Convert a year to epoch time.
    :param year: The year to convert
    :return: The epoch time
    """
    date_str = f"01-01-{year} 00:00:00"
    dt = datetime.datetime.strptime(date_str, "%d-%m-%Y %H:%M:%S")
    epoch_time = int(time.mktime(dt.timetuple()) * 1000)
    return epoch_time
