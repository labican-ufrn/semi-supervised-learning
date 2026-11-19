from enum import Enum

class ThemeColor(tuple, Enum):
    # Cores de fundo
    BACKGROUND = ("#F3F4F6", "#121820")
    SURFACE = ("#FFFFFF", "#1E293B")
    INPUT = ("#F1F5F9", "#273548")
    BORDER = ("#CBD5E1", "#334155")

    # Cores principais
    PRIMARY = ("#2563EB", "#2563EB")
    PRIMARY_HOVER = ("#1D4ED8", "#3B82F6")
    TITLE = ("#2563EB", "#60A5FA")

    # Textos
    TEXT = ("#111827", "#E2E8F0")
    SUBTITLE = ("#64748B", "#94A3B8")

    # Mensagens
    DANGER = ("#DC2626", "#F87171")
    SUCCESS = ("#16A34A", "#4ADE80")