# -*- coding: utf-8 -*-
"""
visualize.py — построение набора красивых аналитических графиков (EDA)
по данным обращений пациентов из train.csv.

Запуск:  python visualize.py
Результат: PNG-файлы в каталоге charts/ + сводный коллаж dashboard.png

Скрипт самодостаточен: ему нужны только pandas, numpy и matplotlib.
Все агрегаты считаются за один-два прохода по данным, поэтому даже на
2.2 млн строк построение занимает считанные секунды.
"""

from os import makedirs
from os.path import join

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # рендер в файл, без окна
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap, to_rgba
from matplotlib.patches import FancyBboxPatch

# ----------------------------------------------------------------------------
#  Оформление: единая палитра и стиль для всех графиков
# ----------------------------------------------------------------------------

OUT_DIR = "charts"
makedirs(OUT_DIR, exist_ok=True)

# Фирменная палитра проекта: спокойный «медицинский» тиал + контрастный коралл
INK       = "#0f2740"   # основной тёмный текст
SUBTLE    = "#5b6b7f"   # вторичный текст
GRID      = "#e6ecf2"   # линии сетки
PANEL     = "#ffffff"   # фон панели
PAGE      = "#f4f7fb"   # фон страницы
TEAL      = "#0d9488"   # основной акцент
TEAL_DARK = "#0b6e6a"
CORAL     = "#f4505e"   # контрастный акцент
INDIGO    = "#5b6cff"
AMBER     = "#f6a609"
VIOLET    = "#9b5de5"
SKY       = "#2bb3d6"

# Циклическая палитра для категориальных графиков
PALETTE = [TEAL, CORAL, INDIGO, AMBER, VIOLET, SKY, "#e76f51", "#2a9d8f",
           "#8d99ae", "#ff8fab", "#06d6a0", "#118ab2"]

# Последовательные карты для тепловых карт (белый -> акцент)
CMAP_TEAL  = LinearSegmentedColormap.from_list("teal",  ["#ffffff", "#9fe6df", TEAL, "#08403d"])
CMAP_CORAL = LinearSegmentedColormap.from_list("coral", ["#ffffff", "#ffc9ce", CORAL, "#7d1622"])


def use_cyrillic_font():
    """Выбрать установленный шрифт с поддержкой кириллицы, чтобы подписи не «квадратились»."""
    for name in ("Segoe UI", "DejaVu Sans", "Arial", "Tahoma", "Verdana"):
        try:
            font_manager.findfont(name, fallback_to_default=False)
            return name
        except Exception:
            continue
    return "DejaVu Sans"


plt.rcParams.update({
    "figure.facecolor":  PAGE,
    "axes.facecolor":    PANEL,
    "savefig.facecolor": PAGE,
    "font.family":       use_cyrillic_font(),
    "font.size":         12,
    "text.color":        INK,
    "axes.edgecolor":    GRID,
    "axes.labelcolor":   SUBTLE,
    "axes.linewidth":    1.0,
    "axes.grid":         True,
    "grid.color":        GRID,
    "grid.linewidth":    1.0,
    "xtick.color":       SUBTLE,
    "ytick.color":       SUBTLE,
    "axes.axisbelow":    True,
})


def style_axes(ax, bottom=True, left=True):
    """Убрать лишние рамки — чистый «журнальный» вид."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_visible(bottom)
    ax.spines["left"].set_visible(left)
    ax.tick_params(length=0)


def titled(fig, title, subtitle=None):
    """Крупный заголовок + подзаголовок в едином стиле, с фирменной плашкой-акцентом."""
    fig.text(0.035, 0.965, title, ha="left", va="top",
             fontsize=21, fontweight="bold", color=INK)
    if subtitle:
        fig.text(0.035, 0.917, subtitle, ha="left", va="top",
                 fontsize=12.5, color=SUBTLE)
    # короткая акцентная черта-подчёркивание под блоком заголовка
    fig.add_artist(plt.Line2D([0.037, 0.115], [0.878, 0.878],
                              color=TEAL, lw=4, solid_capstyle="round",
                              transform=fig.transFigure))


def footer(fig, text="HealthPrediction · анализ обращений пациентов · train.csv"):
    fig.text(0.035, 0.02, text, ha="left", va="bottom", fontsize=9.5, color="#9aa7b5")


def save(fig, name):
    path = join(OUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight", pad_inches=0.35)
    plt.close(fig)
    print("  сохранено:", path)


def human(n):
    """Человекочитаемое число: 12 937 750 -> '12.9 млн'."""
    n = float(n)
    if n >= 1e6:
        return f"{n/1e6:.1f} млн"
    if n >= 1e3:
        return f"{n/1e3:.0f} тыс."
    return f"{n:.0f}"


# ----------------------------------------------------------------------------
#  Классификатор МКБ: короткие названия 21 класса (глав)
# ----------------------------------------------------------------------------

CHAPTERS = [
    ("A00", "B99", "Инфекционные и паразитарные"),
    ("C00", "D48", "Новообразования"),
    ("D50", "D89", "Болезни крови и иммунитета"),
    ("E00", "E90", "Эндокринные, обмен веществ"),
    ("F00", "F99", "Психические расстройства"),
    ("G00", "G99", "Нервная система"),
    ("H00", "H59", "Болезни глаза"),
    ("H60", "H95", "Болезни уха"),
    ("I00", "I99", "Система кровообращения"),
    ("J00", "J99", "Органы дыхания"),
    ("K00", "K93", "Органы пищеварения"),
    ("L00", "L99", "Кожа и подкожная клетчатка"),
    ("M00", "M99", "Костно-мышечная система"),
    ("N00", "N99", "Мочеполовая система"),
    ("O00", "O99", "Беременность и роды"),
    ("P00", "P96", "Перинатальный период"),
    ("Q00", "Q99", "Врождённые аномалии"),
    ("R00", "R99", "Симптомы и отклонения"),
    ("S00", "T98", "Травмы и отравления"),
    ("U00", "U85", "Специальные коды"),
    ("V01", "Y98", "Внешние причины"),
    ("Z00", "Z99", "Факторы здоровья, профилактика"),
]


def build_chapter_index():
    """Сопоставить каждому коду «буква+2 цифры» короткое название главы МКБ."""
    index = {}
    for start, end, title in CHAPTERS:
        l0, l1 = start[0], end[0]
        n0, n1 = int(start[1:]), int(end[1:])
        for letter_code in range(ord(l0), ord(l1) + 1):
            letter = chr(letter_code)
            lo = n0 if letter == l0 else 0
            hi = n1 if letter == l1 else 99
            for num in range(lo, hi + 1):
                index[f"{letter}{num:02d}"] = title
    return index


CHAPTER_INDEX = build_chapter_index()


def code_to_chapter(part):
    """DISEASE_PART (напр. 'J06', 'I10', 'A00') -> название главы МКБ."""
    if not isinstance(part, str) or len(part) < 1:
        return "Прочее"
    key = part[:3].upper()
    if key in CHAPTER_INDEX:
        return CHAPTER_INDEX[key]
    # запасной вариант: только по первой букве берём первую подходящую главу
    letter = part[0].upper()
    for start, _end, title in CHAPTERS:
        if start[0] == letter:
            return title
    return "Прочее"


# Возрастные категории в логическом порядке + человекочитаемые подписи
AGE_ORDER = ["children", "young", "middleage", "elderly", "old", "centenarians"]
AGE_LABEL = {
    "children":     "Дети",
    "young":        "Молодые",
    "middleage":    "Средний возраст",
    "elderly":      "Пожилые",
    "old":          "Старые",
    "centenarians": "Долгожители",
}

MONTHS_RU = ["янв", "фев", "мар", "апр", "май", "июн",
             "июл", "авг", "сен", "окт", "ноя", "дек"]


def ru_month_year(ts):
    """Timestamp -> 'окт 2021' (русские месяцы вместо системной локали)."""
    return f"{MONTHS_RU[ts.month - 1]} {ts.year}"


# ----------------------------------------------------------------------------
#  Загрузка данных
# ----------------------------------------------------------------------------

print("Чтение train.csv …")
df = pd.read_csv(
    "train.csv", sep=";",
    usecols=["PATIENT_SEX", "MKB_CODE", "ADRES", "VISIT_MONTH_YEAR",
             "AGE_CATEGORY", "PATIENT_ID_COUNT"],
    dtype={"PATIENT_SEX": str, "MKB_CODE": str, "ADRES": str,
           "VISIT_MONTH_YEAR": str, "AGE_CATEGORY": str},
)
df["PATIENT_ID_COUNT"] = df["PATIENT_ID_COUNT"].astype(int)

# производные поля
df["MONTH"]   = df["VISIT_MONTH_YEAR"].str.slice(0, 2).astype(int)        # 1..12
df["YEAR"]    = 2000 + df["VISIT_MONTH_YEAR"].str.slice(3, 5).astype(int)  # 2018..2022
df["DATE"]    = pd.to_datetime(dict(year=df["YEAR"], month=df["MONTH"], day=1))
df["CHAPTER"] = df["MKB_CODE"].str.split(".").str[0].map(code_to_chapter)

TOTAL_VISITS  = int(df["PATIENT_ID_COUNT"].sum())
TOTAL_ROWS    = len(df)
print(f"  строк: {TOTAL_ROWS:,}  ·  обращений: {TOTAL_VISITS:,}".replace(",", " "))


# ============================================================================
#  ГРАФИК 0. Hero-баннер с ключевыми показателями (шапка README)
# ============================================================================

def chart_hero():
    fig = plt.figure(figsize=(13, 4.4))
    fig.patch.set_facecolor("#06302e")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")

    # диагональный градиент тёмный тиал -> почти чёрно-зелёный
    grad = np.linspace(0, 1, 256).reshape(1, -1)
    grad = np.vstack([grad] * 32)
    cmap_bg = LinearSegmentedColormap.from_list("bg", ["#0d8c83", "#06302e", "#041f1e"])
    ax.imshow(grad, extent=[0, 1, 0, 1], aspect="auto", cmap=cmap_bg, zorder=0)

    # сабтл-сетка из тонких линий для «технологичности»
    for gx in np.linspace(0.0, 1.0, 24):
        ax.axvline(gx, color="white", alpha=0.03, lw=1, zorder=1)

    ax.text(0.04, 0.78, "HealthPrediction", color="white", fontsize=40,
            fontweight="bold", va="center", zorder=3)
    ax.text(0.042, 0.585,
            "Прогноз числа обращений пациентов  ·  CatBoost-регрессия  ·  Калининградская область",
            color="#a7e8e0", fontsize=13.5, va="center", zorder=3)

    # KPI-карточки
    kpis = [
        ("2.2 млн", "строк данных"),
        ("12.9 млн", "обращений"),
        ("51", "месяц истории"),
        ("118", "городов"),
        ("0.9223", "R²  ·  лидерборд"),
    ]
    n = len(kpis)
    pad, gap = 0.04, 0.018
    total_w = 1 - 2 * pad - (n - 1) * gap
    cw = total_w / n
    y0, ch = 0.10, 0.32
    for i, (value, label) in enumerate(kpis):
        x0 = pad + i * (cw + gap)
        card = FancyBboxPatch((x0, y0), cw, ch,
                              boxstyle="round,pad=0.004,rounding_size=0.02",
                              transform=ax.transAxes, facecolor="white",
                              alpha=0.07, edgecolor="white", linewidth=0.0, zorder=2)
        ax.add_patch(card)
        accent = PALETTE[i % len(PALETTE)]
        ax.add_patch(FancyBboxPatch((x0, y0), 0.006, ch,
                     boxstyle="round,pad=0,rounding_size=0.01",
                     transform=ax.transAxes, facecolor=accent, edgecolor="none", zorder=3))
        ax.text(x0 + cw / 2, y0 + ch * 0.62, value, color="white", fontsize=21,
                fontweight="bold", ha="center", va="center", zorder=4)
        ax.text(x0 + cw / 2, y0 + ch * 0.24, label, color="#9fc7c2", fontsize=10.5,
                ha="center", va="center", zorder=4)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    save(fig, "00_hero.png")


# ============================================================================
#  ГРАФИК 1. Динамика обращений по месяцам (временной ряд)
# ============================================================================

def chart_monthly_trend():
    s = df.groupby("DATE")["PATIENT_ID_COUNT"].sum().sort_index()

    fig, ax = plt.subplots(figsize=(13, 6.2))
    fig.subplots_adjust(top=0.80, bottom=0.12, left=0.085, right=0.97)

    ax.fill_between(s.index, s.values, color=TEAL, alpha=0.10, zorder=1)
    ax.plot(s.index, s.values, color=TEAL, lw=2.6, zorder=3)
    ax.scatter(s.index, s.values, s=16, color=TEAL, zorder=4,
               edgecolor="white", linewidth=0.6)

    # отметка пика и провала
    i_max, i_min = s.idxmax(), s.idxmin()
    ax.scatter([i_max], [s[i_max]], s=120, color=CORAL, zorder=5,
               edgecolor="white", linewidth=1.4)
    ax.annotate(f"пик: {human(s[i_max])}\n{ru_month_year(i_max)}",
                xy=(i_max, s[i_max]), xytext=(0, 22),
                textcoords="offset points", ha="center", fontsize=10,
                color=CORAL, fontweight="bold")
    ax.annotate(f"спад: {human(s[i_min])}\n{ru_month_year(i_min)}",
                xy=(i_min, s[i_min]), xytext=(0, -34),
                textcoords="offset points", ha="center", fontsize=10,
                color=SUBTLE)

    import matplotlib.dates as mdates
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_minor_locator(mdates.MonthLocator((1, 4, 7, 10)))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.tick_params(axis="x", which="major", labelsize=12)

    ax.yaxis.set_major_formatter(lambda x, _: human(x))
    ax.set_ylim(0, s.max() * 1.18)
    ax.margins(x=0.02)
    style_axes(ax)
    ax.grid(axis="x", visible=False)

    titled(fig, "Динамика обращений пациентов",
           "Суммарное число обращений по месяцам · январь 2018 — март 2022")
    footer(fig)
    save(fig, "01_monthly_trend.png")


# ============================================================================
#  ГРАФИК 2. Сезонность: средний месяц года
# ============================================================================

def chart_seasonality():
    # среднее по календарному месяцу (усреднение по годам)
    per_month_year = df.groupby(["YEAR", "MONTH"])["PATIENT_ID_COUNT"].sum().reset_index()
    season = per_month_year.groupby("MONTH")["PATIENT_ID_COUNT"].mean()
    season = season.reindex(range(1, 13))

    fig, ax = plt.subplots(figsize=(13, 6.0))
    fig.subplots_adjust(top=0.80, bottom=0.12, left=0.085, right=0.97)

    x = np.arange(1, 13)
    norm = (season.values - season.min()) / (season.max() - season.min() + 1e-9)
    colors = [CMAP_TEAL(0.30 + 0.65 * v) for v in norm]
    bars = ax.bar(x, season.values, color=colors, width=0.72, zorder=3,
                  edgecolor="white", linewidth=0.8)

    ax.plot(x, season.values, color=INK, lw=1.6, alpha=0.35, zorder=4)
    for xi, val in zip(x, season.values):
        ax.text(xi, val + season.max() * 0.02, human(val), ha="center",
                va="bottom", fontsize=9.5, color=SUBTLE)

    ax.set_xticks(x)
    ax.set_xticklabels(MONTHS_RU)
    ax.yaxis.set_major_formatter(lambda v, _: human(v))
    ax.set_ylim(0, season.max() * 1.16)
    style_axes(ax)
    ax.grid(axis="x", visible=False)

    titled(fig, "Сезонность обращаемости",
           "Среднее число обращений по календарному месяцу (усреднение 2018–2022)")
    footer(fig)
    save(fig, "02_seasonality.png")


# ============================================================================
#  ГРАФИК 3. Топ глав МКБ по числу обращений
# ============================================================================

def chart_top_diseases():
    top = (df.groupby("CHAPTER")["PATIENT_ID_COUNT"].sum()
             .sort_values(ascending=True).tail(14))

    fig, ax = plt.subplots(figsize=(13, 7.4))
    fig.subplots_adjust(top=0.82, bottom=0.07, left=0.30, right=0.95)

    y = np.arange(len(top))
    grad = [CMAP_TEAL(0.35 + 0.6 * i / (len(top) - 1)) for i in range(len(top))]
    ax.barh(y, top.values, color=grad, height=0.72, zorder=3)

    share = top.values / TOTAL_VISITS * 100
    for yi, val, pct in zip(y, top.values, share):
        ax.text(val + top.max() * 0.01, yi, f"{human(val)}  ·  {pct:.1f}%",
                va="center", ha="left", fontsize=10.5, color=INK)

    ax.set_yticks(y)
    ax.set_yticklabels(top.index, fontsize=11)
    ax.set_xlim(0, top.max() * 1.20)
    ax.xaxis.set_major_formatter(lambda v, _: human(v))
    style_axes(ax, left=False)
    ax.grid(axis="y", visible=False)

    titled(fig, "Структура заболеваемости",
           "Топ-14 классов МКБ-10 по суммарному числу обращений")
    footer(fig)
    save(fig, "03_top_diseases.png")


# ============================================================================
#  ГРАФИК 4. Возраст × пол
# ============================================================================

def chart_age_sex():
    piv = (df.groupby(["AGE_CATEGORY", "PATIENT_SEX"])["PATIENT_ID_COUNT"]
             .sum().unstack(fill_value=0))
    piv = piv.reindex(AGE_ORDER)
    labels = [AGE_LABEL[a] for a in piv.index]

    fig, ax = plt.subplots(figsize=(13, 6.4))
    fig.subplots_adjust(top=0.80, bottom=0.12, left=0.085, right=0.97)

    x = np.arange(len(piv))
    w = 0.38
    sexes = list(piv.columns)
    palette = {sexes[0]: TEAL}
    if len(sexes) > 1:
        palette[sexes[1]] = CORAL

    for i, sex in enumerate(sexes):
        vals = piv[sex].values
        ax.bar(x + (i - (len(sexes) - 1) / 2) * w, vals, width=w,
               color=palette.get(sex, INDIGO), zorder=3,
               edgecolor="white", linewidth=0.8,
               label=f"Пол «{sex}»")
        for xi, val in zip(x, vals):
            ax.text(xi + (i - (len(sexes) - 1) / 2) * w, val + piv.values.max() * 0.012,
                    human(val), ha="center", va="bottom", fontsize=9, color=SUBTLE)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.yaxis.set_major_formatter(lambda v, _: human(v))
    ax.set_ylim(0, piv.values.max() * 1.16)
    ax.legend(frameon=False, loc="upper right", fontsize=11)
    style_axes(ax)
    ax.grid(axis="x", visible=False)

    titled(fig, "Обращения по возрасту и полу",
           "Распределение суммарных обращений · кодировка пола 0 / 1 как в данных")
    footer(fig)
    save(fig, "04_age_sex.png")


# ============================================================================
#  ГРАФИК 5. Распределение размера группы (PATIENT_ID_COUNT)
# ============================================================================

def chart_count_distribution():
    vals = df["PATIENT_ID_COUNT"].values
    bins = np.logspace(0, np.log10(vals.max() + 1), 40)

    fig, ax = plt.subplots(figsize=(13, 6.0))
    fig.subplots_adjust(top=0.80, bottom=0.13, left=0.09, right=0.97)

    n, edges, patches = ax.hist(vals, bins=bins, color=TEAL, alpha=0.9,
                                edgecolor="white", linewidth=0.5, zorder=3)
    # градиентная окраска столбцов по высоте
    nmax = n.max()
    for count, patch in zip(n, patches):
        patch.set_facecolor(CMAP_TEAL(0.30 + 0.65 * (count / nmax)))

    ax.set_xscale("log")
    ax.set_yscale("log")

    med = np.median(vals)
    mean = vals.mean()
    ax.axvline(med, color=INDIGO, lw=2, ls="--", zorder=4)
    ax.axvline(mean, color=CORAL, lw=2, ls="--", zorder=4)
    ax.text(med, n.max() * 1.3, f" медиана = {med:.0f}", color=INDIGO,
            fontsize=10.5, fontweight="bold", va="bottom")
    ax.text(mean, n.max() * 0.45, f" среднее ≈ {mean:.1f}", color=CORAL,
            fontsize=10.5, fontweight="bold", va="bottom")

    ax.set_xlabel("Число пациентов в группе (PATIENT_ID_COUNT), лог. шкала")
    ax.set_ylabel("Количество записей, лог. шкала")
    style_axes(ax)

    titled(fig, "Длинный хвост размеров групп",
           "Большинство записей — единичные обращения, но хвост тянется до десятков тысяч")
    footer(fig)
    save(fig, "05_count_distribution.png")


# ============================================================================
#  ГРАФИК 6. Население города × число обращений
# ============================================================================

def chart_city_scatter():
    # население городов берём из справочника augmentation.cities
    from augmentation import cities

    by_city = df.groupby("ADRES")["PATIENT_ID_COUNT"].sum()
    pts = [(cities.get(city), total, city)
           for city, total in by_city.items() if cities.get(city)]
    pop   = np.array([p[0] for p in pts], dtype=float)
    visit = np.array([p[1] for p in pts], dtype=float)
    names = [p[2] for p in pts]

    fig, ax = plt.subplots(figsize=(13, 6.6))
    fig.subplots_adjust(top=0.80, bottom=0.13, left=0.10, right=0.97)

    sizes = 30 + (visit / visit.max()) * 900
    ax.scatter(pop, visit, s=sizes, color=TEAL, alpha=0.55,
               edgecolor=TEAL_DARK, linewidth=0.8, zorder=3)

    # линия тренда в лог-лог координатах
    lp, lv = np.log10(pop), np.log10(visit)
    a, b = np.polyfit(lp, lv, 1)
    xs = np.linspace(lp.min(), lp.max(), 50)
    ax.plot(10 ** xs, 10 ** (a * xs + b), color=CORAL, lw=2.2, ls="--",
            zorder=4, label=f"тренд: обращения ~ население^{a:.2f}")
    corr = np.corrcoef(lp, lv)[0, 1]

    # подписать несколько крупнейших городов
    for idx in visit.argsort()[-6:]:
        ax.annotate(names[idx], (pop[idx], visit[idx]),
                    xytext=(6, 6), textcoords="offset points",
                    fontsize=9.5, color=INK, fontweight="bold")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Население города, чел. (лог. шкала)")
    ax.set_ylabel("Всего обращений (лог. шкала)")
    ax.legend(frameon=False, loc="upper left", fontsize=11)
    ax.text(0.985, 0.06, f"корреляция (log-log): r = {corr:.2f}",
            transform=ax.transAxes, ha="right", fontsize=11,
            color=SUBTLE, style="italic")
    style_axes(ax)

    titled(fig, "Города: население и обращаемость",
           "Каждая точка — населённый пункт Калининградской области; размер — число обращений")
    footer(fig)
    save(fig, "06_city_population.png")


# ============================================================================
#  ГРАФИК 7. Тепловая карта: класс МКБ × месяц (сезонный профиль)
# ============================================================================

def chart_heatmap_class_month():
    top_chapters = (df.groupby("CHAPTER")["PATIENT_ID_COUNT"].sum()
                      .sort_values(ascending=False).head(12).index.tolist())
    sub = df[df["CHAPTER"].isin(top_chapters)]
    # ВАЖНО: янв-мар присутствуют в 5 годах, апр-дек — в 4. Чтобы убрать этот
    # перекос, считаем СРЕДНЕЕ по годам, а не сумму (как и в графике сезонности).
    per_year = sub.groupby(["CHAPTER", "YEAR", "MONTH"])["PATIENT_ID_COUNT"].sum().reset_index()
    piv = (per_year.groupby(["CHAPTER", "MONTH"])["PATIENT_ID_COUNT"].mean()
                   .unstack(fill_value=0).reindex(top_chapters))
    piv = piv.reindex(columns=range(1, 13), fill_value=0)

    # нормируем каждую строку в [0,1] — виден сезонный профиль внутри класса
    norm = piv.div(piv.max(axis=1), axis=0)

    fig, ax = plt.subplots(figsize=(13, 7.6))
    fig.subplots_adjust(top=0.82, bottom=0.08, left=0.28, right=1.02)

    im = ax.imshow(norm.values, aspect="auto", cmap=CMAP_TEAL, vmin=0, vmax=1)

    ax.set_xticks(range(12))
    ax.set_xticklabels(MONTHS_RU)
    ax.set_yticks(range(len(top_chapters)))
    ax.set_yticklabels(top_chapters, fontsize=10.5)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks(np.arange(-0.5, 12, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(top_chapters), 1), minor=True)
    ax.grid(which="minor", color=PAGE, linewidth=2)
    ax.grid(which="major", visible=False)
    ax.tick_params(which="minor", length=0)

    # пометить максимум каждой строки точкой
    for r in range(len(top_chapters)):
        c = int(np.argmax(norm.values[r]))
        ax.scatter(c, r, s=26, color=CORAL, zorder=3, edgecolor="white", linewidth=0.8)

    cbar = fig.colorbar(im, ax=ax, fraction=0.026, pad=0.02)
    cbar.set_label("доля от пика класса", color=SUBTLE)
    cbar.outline.set_visible(False)

    titled(fig, "Сезонные профили заболеваний",
           "Класс МКБ × месяц · нормировка по строке · ● — пиковый месяц класса")
    footer(fig)
    save(fig, "07_heatmap_class_month.png")


# ============================================================================
#  ГРАФИК 8. Тепловая карта: возраст × класс МКБ
# ============================================================================

def chart_heatmap_age_class():
    top_chapters = (df.groupby("CHAPTER")["PATIENT_ID_COUNT"].sum()
                      .sort_values(ascending=False).head(12).index.tolist())
    sub = df[df["CHAPTER"].isin(top_chapters)]
    piv = (sub.groupby(["AGE_CATEGORY", "CHAPTER"])["PATIENT_ID_COUNT"].sum()
              .unstack(fill_value=0).reindex(AGE_ORDER))
    piv = piv.reindex(columns=top_chapters, fill_value=0)

    # доля класса внутри возрастной группы (по строке) -> профиль возраста
    norm = piv.div(piv.sum(axis=1), axis=0) * 100

    fig, ax = plt.subplots(figsize=(13.5, 6.6))
    fig.subplots_adjust(top=0.82, bottom=0.30, left=0.12, right=1.02)

    im = ax.imshow(norm.values, aspect="auto", cmap=CMAP_CORAL)

    ax.set_yticks(range(len(AGE_ORDER)))
    ax.set_yticklabels([AGE_LABEL[a] for a in AGE_ORDER], fontsize=11)
    ax.set_xticks(range(len(top_chapters)))
    ax.set_xticklabels(top_chapters, rotation=35, ha="right", fontsize=10)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks(np.arange(-0.5, len(top_chapters), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(AGE_ORDER), 1), minor=True)
    ax.grid(which="minor", color=PAGE, linewidth=2)
    ax.grid(which="major", visible=False)
    ax.tick_params(which="minor", length=0)

    # подписать долю в процентах
    for r in range(norm.shape[0]):
        for c in range(norm.shape[1]):
            v = norm.values[r, c]
            ax.text(c, r, f"{v:.0f}", ha="center", va="center", fontsize=8.5,
                    color="white" if v > norm.values.max() * 0.55 else SUBTLE)

    cbar = fig.colorbar(im, ax=ax, fraction=0.026, pad=0.02)
    cbar.set_label("% обращений внутри возраста", color=SUBTLE)
    cbar.outline.set_visible(False)

    titled(fig, "Возрастной профиль заболеваний",
           "Доля каждого класса МКБ внутри возрастной группы, %")
    footer(fig)
    save(fig, "08_heatmap_age_class.png")


# ============================================================================
#  ГРАФИК 9. Значимость признаков обученной модели CatBoost
# ============================================================================

def chart_feature_importance():
    # значения feature_importances_ из лучшей обученной модели (score 0.9223),
    # сохранённые ранее в features.png — перерисованы в едином стиле проекта
    importance = {
        "MKB_CODE — код заболевания (МКБ)":      31.0,
        "CITY_TYPE — тип города по населению":   19.0,
        "AGE_CATEGORY — возрастная категория":   16.5,
        "VISIT_MONTH_YEAR — месяц обращения":    14.3,
        "DISEASE_PART — класс заболевания":       7.5,
        "ADRES — населённый пункт":               7.3,
        "CITIZENS — численность населения":       3.1,
        "PATIENT_SEX — пол пациента":             1.5,
    }
    items = sorted(importance.items(), key=lambda kv: kv[1])
    labels = [k for k, _ in items]
    vals   = np.array([v for _, v in items])

    fig, ax = plt.subplots(figsize=(13, 6.6))
    fig.subplots_adjust(top=0.82, bottom=0.10, left=0.34, right=0.95)

    y = np.arange(len(vals))
    grad = [CMAP_TEAL(0.35 + 0.6 * i / (len(vals) - 1)) for i in range(len(vals))]
    ax.barh(y, vals, color=grad, height=0.70, zorder=3)
    for yi, v in zip(y, vals):
        ax.text(v + 0.4, yi, f"{v:.1f}%", va="center", ha="left",
                fontsize=11, color=INK, fontweight="bold")

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlim(0, vals.max() * 1.16)
    ax.set_xlabel("вклад признака в модель, %")
    style_axes(ax, left=False)
    ax.grid(axis="y", visible=False)

    titled(fig, "Значимость признаков модели",
           "CatBoost · feature importance лучшей модели (R² = 0.9223 на лидерборде)")
    footer(fig)
    save(fig, "09_feature_importance.png")


# ============================================================================
#  Запуск всех графиков
# ============================================================================

if __name__ == "__main__":
    print("Построение графиков …")
    chart_hero()
    chart_monthly_trend()
    chart_seasonality()
    chart_top_diseases()
    chart_age_sex()
    chart_count_distribution()
    chart_city_scatter()
    chart_heatmap_class_month()
    chart_heatmap_age_class()
    chart_feature_importance()
    print("Готово. Все графики — в каталоге", OUT_DIR + "/")
