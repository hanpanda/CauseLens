import pandas as pd
import datetime
import pytz


def get_current_time() -> str:
    return pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')    


def to_timestamp(datetime_str: str, from_tz_str: str='Asia/Shanghai', unit: str='s') -> int:
    """将日期时间字符串转换为UTC时间戳。

    :param datetime_str: 日期时间字符串
    :param from_tz_str: 输入日期时间的时区字符串，默认为'Asia/Shanghai'（北京时间）
    :param unit: 时间戳的单位，默认为秒('s')，可选毫秒('ms')
    :return: UTC时间戳
    """
    try:
        # 加载时区信息
        from_tz = pytz.timezone(from_tz_str)
    except pytz.UnknownTimeZoneError:
        raise ValueError(f"Invalid timezone string: {from_tz_str}")

    try:
        # 将字符串转换为指定时区的datetime对象
        dt = datetime.datetime.strptime(datetime_str, '%Y-%m-%d %H:%M:%S')
        dt = from_tz.localize(dt)
    except ValueError:
        raise ValueError("Invalid datetime string format. Expected '%Y-%m-%d %H:%M:%S'.")

    # 转换到UTC
    dt_utc = dt.astimezone(pytz.utc)
    
    # 转换为时间戳
    timestamp = int(dt_utc.timestamp())
    
    # 如果需要返回的单位是毫秒
    if unit == 'ms':
        timestamp *= 1000
    elif unit != 's':
        raise ValueError("Invalid unit. Expected 's' or 'ms'.")

    return timestamp


def to_datetime(timestamp: int, to_tz_str: str='Asia/Shanghai', unit: str='s') -> str:
    """
    将时间戳转换为指定时区的日期时间字符串。

    :param timestamp: int, 时间戳，如果单位是秒，则为自1970年1月1日以来的秒数；
                      如果单位是毫秒，则为自1970年1月1日以来的毫秒数。
    :param to_tz_str: str, 目标时区的字符串表示，默认为'Asia/Shanghai'。
    :param unit: str, 时间戳的单位，'s'表示秒，'ms'表示毫秒，默认为's'。
    :return: str, 格式化的日期时间字符串。
    """
    # 根据时间戳单位转换时间戳为秒
    if unit == 'ms':
        timestamp /= 1000  # 如果单位是毫秒，则除以1000转换为秒
    
    # 将时间戳转换为UTC时区的datetime对象
    utc_datetime = datetime.datetime.fromtimestamp(timestamp, tz=datetime.timezone.utc)
    
    # 获取目标时区对象
    to_tz = pytz.timezone(to_tz_str)
    
    # 将UTC datetime对象转换为指定时区的datetime对象
    localized_datetime = utc_datetime.replace(tzinfo=pytz.utc).astimezone(to_tz)
    
    # 返回格式化的日期时间字符串
    return localized_datetime.strftime('%Y-%m-%d %H:%M:%S')