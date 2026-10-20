import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("LOBS_GUI_NO_SETTINGS", "1")  # tests never read or write the user layout
