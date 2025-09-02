import telegram, httpx
print("telegram module path:", telegram.__file__)
import telegram.ext as ext
print("PTB Application exists:", hasattr(ext, "Application"))
import httpx as h
print("httpx:", h.__version__)
