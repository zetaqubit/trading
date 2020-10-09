from dataclasses import dataclass

@dataclass
class DateRange:
  """Specifies a range of dates, using [start, end)."""
  start: str = None
  end: str = None
