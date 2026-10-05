# -*- coding: utf-8 -*-
"""Build the IAD Lab 1 report.

Writes report.json (content artifact) and renders the .docx in the house
style of the report-maker skill (TNR 14 pt, justified, bold section labels),
extended with:
  - **inline bold** markup inside text blocks
  - "section" blocks: bold label lines (house style: bold label + colon,
    never heading styles)
  - short paragraphs instead of text walls
"""
import json
import os
import re
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.opc.constants import RELATIONSHIP_TYPE as RT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt

BASE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(BASE, "figures")
OUT_DIR = os.path.join(BASE, "report")
OUT_DOCX = os.path.join(OUT_DIR, "ІАД_КН-2327Б_EDA_ЛР-1.docx")

BODY_FONT = "Times New Roman"
CODE_FONT = "Cascadia Mono"
BODY_SIZE = Pt(14)
CODE_SIZE = Pt(9.5)
FIG_WIDTH = Cm(14.5)
MARGIN_CM = 1.5


def f(name):
    return os.path.join(FIG, name)


# ---------------------------------------------------------------------------
# Content
# ---------------------------------------------------------------------------
META = {
    "course_code": "ІАД",
    "course_name": "Інтелектуальний аналіз даних",
    "lab_number": 1,
    "topic": "Аналіз та візуалізація даних (EDA)",
    "author": "Багрія-Белз Андрій",
    "group": "КН-2327Б",
    "objective": "Опанувати методи дослідження даних (EDA) та побудову діаграм за допомогою Python (pandas, matplotlib, seaborn).",
    "title_word": "ЗВІТ",
    "title_style": "standard",
}



INTRO = [
    "Метою лабораторної роботи є проведення дослідницького (EDA) аналізу даних та побудова діаграм у Python за допомогою бібліотек **pandas**, **matplotlib** та **seaborn**.",
    "Робота складається з двох частин. У **першій** аналізуємо тестовий датасет **flats.csv** — дані про нерухомість (місто, кількість кімнат, площа, ціна), який містить «брудні» значення та потребує попереднього очищення. У **другій** аналізуємо реальний датасет **Penguin Species** (Palmer Penguins) з бібліотеки seaborn, демонструючи різні типи діаграм, зокрема обов'язковий **boxplot**.",
]

QA = [
    "**1. Розміри dataframe:** **839 рядків × 4 стовпці**. На етапі технічного завантаження та перевірки наявності пустих значень (NaN) збережено всі 839 рядків, оскільки формальних пропусків у стовпцях не було. Подальша фільтрація стосувалася вилучення змістовних цінових аномалій (оренда/USD), що детально пояснено у відповідному розділі.",
    "**2. Перші 6, перші 15 та останні 6 рядків:** успішно виведено за допомогою методів head(6), head(15) та tail(6). Початкові записи представляють квартири у Вінниці (площа 31.0–120.0 м², ціна 562 500 – 1 875 000 грн), а завершальні записи вибірки — об'єкти у Хмельницькому (площа 35.58–60.00 м², ціна 212 500 – 522 500 грн).",
    "**3. Назви стовпців:** «Місто», «Кімнат», «Загальна_площа», «Ціна».",
    "**4. Кількість змінних:** **4**.",
    "**5. Кількість унікальних міст:** **13**.",
    "**6. Чи всі це міста?** **Ні.** Значення «Києво-Святошинський» позначає район (РДА), а не місто — таких рядків **19**. Тому реальних міст у датасеті **12**.",
    "**7. 3-кімнатних квартир в Одесі:** **11**.",
    "**8. Медіана площі 1-кімнатної квартири у Львові:** **43.0 м²** (за **7** оголошень).",
]

FLATS_ANALYSIS = [
    "**Джерело та склад даних:** датасет flats.csv містить 839 оголошень про нерухомість в Україні (місто, кількість кімнат, площа, ціна).",
    "**Аудит та очищення цінових аномалій.** У стовпці «Загальна_площа» виправлено кому на крапку та наукову нотацію; пропусків типу NaN не виявлено (технічно завантажено всі 839 рядків). Проте предметний аудит виявив **64 записи** з ціною < 100 000 грн (від 10 200 до 75 000 грн при площах до 222.6 м²), що є артефактами введення цін в USD або ставок місячної оренди замість вартості продажу. На графіках вони утворювали штучну лінію біля нуля, тому для статистичного аналізу ринку купівлі-продажу сформовано валідну вибірку: **N = 775**.",
    "**Узгодження обсягів вибірок (N) між діаграмами.** Різні діаграми оперують специфічними вибірками відповідно до своєї аналітичної мети: на стовпчиковій діаграмі міст відображено **820 оголошень** (вилучено 19 записів району, 839 − 19 = 820); на графіку розсіювання — **775 валідних цін продажу** (вилучено 64 артефакти оренди/USD, 839 − 64 = 775); на boxplot — **774 об'єкти** (квартири від 1 до 5 кімнат, окрім одного 6-кімнатного викиду); на гістограмі цін — **767 квартир** (відсічено 8 екстремальних викидів > 6.0 млн грн на рівні 99-го перцентиля, 775 − 8 = 767, для детального показу основної маси).",
    "**Статистика цін.** Для валідної вибірки продажу (N = 775) середня ціна становить **1 126 812 грн**, а медіана — **825 000 грн** (медіана площі — **53.53 м²**). Середнє помітно вище за медіану: розподіл має виражену правосторонню асиметрію через одиничні дорогі об'єкти до **12,25 млн грн**.",
    "**1. Стовпчикова діаграма міст (Bar chart, N = 820).** Горизонтальна орієнтація забезпечує чітке читання назв усіх 12 міст; діаграма охоплює 820 оголошень (97.7% вибірки) після виключення 19 записів району «Києво-Святошинський».\n**Висновок аналізу (що показав графік):** Ринок у вибірці має різку географічну поляризацію та концентрацію пропозиції. Понад **56% усіх оголошень** припадає лише на два міста: **Вінницю (275 об'єктів, 33.5%)** та **Київ (186 об'єктів, 22.7%)**. Решта 10 обласних центрів представлені незначними частками (від 12 до 68 квартир), а найменша активність зафіксована у Тернополі (12) та Запоріжжі (25). Це свідчить про те, що первинний масив зібрано нерівномірно і він відображає переважно сегмент Вінницького та столичного ринків, що вимагає обережності при загальних узагальненнях.",
    "**2. Гістограма розподілу цін продажу (Histogram, N = 767).** Розподіл обрізано на рівні 99-го перцентиля (**6.0 млн грн**) з кроком біна **0.25 млн грн** для наочного масштабування основної маси житла (8 викидів винесено в анотацію). На осі X позначено межі кожного інтервалу із нулем на початку координат.\n**Висновок аналізу (що показав графік):** Розподіл вартості житла є різко асиметричним із довгою правосторонньою концентрацією. Модальний інтервал становить **0.50–0.75 млн грн** (пік припадає на ~750 тис. грн), що формує ядро платоспроможного попиту. Медіанна ціна (**0.83 млн грн**) відчутно менша за середнє арифметичне (**1.13 млн грн**): розрив у **300 тис. грн (+36%)** спричинений одиничними ультрадорогими об'єктами преміум-сегмента (до 12.25 млн грн). Це наочно доводить, що для аналізу ринку нерухомості медіана є значно більш надійною та об'єктивною мірою, ніж середнє значення.",
    "**3. Scatter-діаграма площа–ціна (N = 775).** Графік побудовано для всіх 775 валідних об'єктів продажу. Використання напівпрозорих маркерів та логарифмічної шкали осі Y усунуло скучення точок біля нуля і дозволило відобразити повний діапазон цін включно з максимальним викидом у 12.25 млн грн. Нанесено експоненційний тренд.\n**Висновок аналізу (що показав графік):** Між площею квартири та її ціною спостерігається стійкий прямий зв'язок: зі збільшенням метражу вартість нелінійно зростає (коефіцієнт кореляції **r = 0.69**). Водночас графік розкриває колосальну цінову дисперсію для об'єктів зі схожою площею: наприклад, у діапазоні **50–70 м²** вартість житла варіюється від **450 тис. грн** до **понад 3.5 млн грн** (майже у 8 разів!). Це доводить, що площа визначає лише нижній базовий поріг вартості, тоді як вирішальну роль відіграють престижність міста (Київ суттєво дорожчий за регіони), категорія новобудови, розташування та якість оздоблення.",
    "**4. Boxplot цін за кількістю кімнат (N = 774).** Досліджено розподіл цін для квартир від 1 до 5 кімнат на логарифмічній шкалі (вилучено 1 запис 6-кімнатної квартири як одиничне спостереження, n = 1).\n**Висновок аналізу (що показав графік):** Кількість кімнат виступає ключовим фактором ступінчастої цінової диференціації житла. Медіанна ціна монотонно зростає з кожною кімнатою: **1-кімнатні — 0.59 млн грн**, **2-кімнатні — 0.84 млн грн (+42%)**, **3-кімнатні — 1.25 млн грн (+49%)**, **4-кімнатні — 1.65 млн грн (+32%)**, **5-кімнатні — 3.00 млн грн (+82%)**. Крім того, графік демонструє різке зростання варіативності (міжквартильного розмаху IQR та довжини верхніх «вусів») зі збільшенням кімнатності: ринок 1- і 2-кімнатних квартир є відносно стандартизованим і компактним, тоді як сегменти 3–5 кімнат характеризуються високим рівнем індивідуалізації, великою часткою елітної нерухомості та статистичними викидами у верхньому ціновому діапазоні.",
]

PENG_DESC = [
    "**Датасет:** Penguin Species (Palmer Penguins) — п. 13 списку датасетів; містить біометричні вимірювання 3 видів пінгвінів (Adelie, Chinstrap, Gentoo) на островах Антарктики (розміри дзьоба, довжина ласт, маса тіла, стать). Джерело: [seaborn-data, penguins.csv](https://github.com/mwaskom/seaborn-data) (офіційний проєкт: [palmerpenguins](https://github.com/mcnakhaee/palmerpenguins)).",
    "**Очищення даних:** у вихідній таблиці 344 записи; 2 рядки з пропущеними значеннями у ключових числових вимірюваннях видалено. Усі 4 діаграми побудовано на єдиній узгодженій вибірці (**строго N = 342**): **Adelie — 151**, **Gentoo — 123**, **Chinstrap — 68**.",
]

PENG_ANALYSIS = [
    "**1. Boxplot (довжина ласт за видами, обов'язковий, N = 342).** Досліджено довжину ласт для трьох видів пінгвінів на узгодженій вибірці (N = 342). Медіани становлять: **Gentoo — 216.0 мм**, **Chinstrap — 196.0 мм**, **Adelie — 190.0 мм**. Для виду Adelie на графіку стрілками та рамками виділено два статистичні викиди — **172 мм** та **210 мм**, які виходять за межі **1.5 · IQR**.\n**Висновок аналізу (що показав графік):** Довжина ласт виступає винятково сильним біометричним маркером міжвидової класифікації. Вид **Gentoo** має кардинально довші ласти (IQR 212–221 мм) і практично без перекриття відокремлений від двох інших видів, що дозволяє безпомилково ідентифікувати особини Gentoo лише за цим виміром. Види **Adelie** та **Chinstrap** є морфологічно ближчими, проте Chinstrap стабільно перевершує Adelie в середньому на 6 мм.",
    "**2. Histogram/KDE (розподіл маси тіла, N = 342).** Діаграму реалізовано у вигляді трьох окремих панелей (faceted panels) зі шкалою густини (**density**) замість абсолютних кількостей, що повністю усуває оптичне накладання груп. На кожну панель нанесено вертикальні лінії медіани (помаранчевий пунктир) та середнього (фіолетова крапка).\n**Висновок аналізу (що показав графік):** Маса тіла демонструє чіткий бімодальний характер популяції. Види **Adelie** (медіана **3 700 г**, середнє **3 701 г**) та **Chinstrap** (медіана **3 700 г**, середнє **3 733 г**) формують легку вагову категорію з симетричним розподілом навколо 3.7 кг. Натомість вид **Gentoo** утворює абсолютно окрему «важку категорію» з медіаною **5 000 г** та середнім **5 076 г** (діапазон 4.0–6.3 кг). Це вказує на еволюційну адаптацію Gentoo для глибоководного пірнання, що потребує значного запасу маси тіла.",
    "**3. Pie-діаграма (розподіл видів, N = 342).** Сектори впорядковано за годинниковою стрілкою від 12:00 за спаданням часток; суму відсотків строго узгоджено до 100.0% алгоритмом найбільших залишків: **Adelie — 151 (44.1%)**, **Gentoo — 123 (36.0%)** та **Chinstrap — 68 (19.9%)**.\n**Висновок аналізу (що показав графік):** Видова структура колонії є помітно незбалансованою: майже половину популяції складає вид **Adelie (44.1%)**, понад третину — **Gentoo (36.0%)**, а вид **Chinstrap є найменш представленим (19.9%)**, поступаючись Adelie більш ніж удвічі. Така диспропорція вимагає обов'язкової стратифікації при подальшому навчанні моделей класифікації, щоб запобігти зміщенню рішень у бік мажоритарного класу.",
    "**4. Scatter-діаграма (маса тіла vs довжина ласт, N = 342).** Напівпрозорі маркери (alpha = 0.65) з індивідуальними фігурами усунули накладання точок; побудовано спільну лінію регресії та окремі прямі для кожного виду.\n**Висновок аналізу (що показав графік):** Графік демонструє потужну позитивну алометричну залежність: більша маса тіла вимагає більшої площі веслувальної поверхні ласт. Загальний коефіцієнт кореляції Пірсона становить **r = 0.871**. Побудова роздільних регресій (**Adelie: r = 0.468**, **Chinstrap: r = 0.642**, **Gentoo: r = 0.703**) нівелює ризик парадоксу Сімпсона й підтверджує наявність стійкого внутрішньовидового зв'язку. Gentoo формує відокремлений кластер у верхньому правому квадранті, тоді як Adelie та Chinstrap утворюють паралельні кластери з дещо різними кутами нахилу.",
]
CONCLUSION = ("На цій лабораторній роботі я ознайомився з основними етапами дослідження даних (EDA): "
              "завантаження, очищення «брудних» даних, описова статистика та побудова діаграм. "
              "Закріпив навички роботи з pandas, matplotlib та seaborn, зокрема побудови scatter, pie, "
              "histogram та boxplot, і навчився інтерпретувати розподіли, кореляції та викиди. "
              "Зрозумів, що правильне попереднє очищення даних є критично важливим для коректних статистичних оцінок.")

BODY = [
    {"type": "text", "text": INTRO[0]},
    {"type": "text", "text": INTRO[1]},

    {"type": "task", "number": 1,
     "text": "Завантажити та очистити датасет flats.csv, відповісти на контрольні запитання методички та побудувати 4 діаграми (стовпчикову за містами, гістограму цін, графік розсіювання площа–ціна та коробчату діаграму цін за кількістю кімнат)."},
    {"type": "result", "label": "Результати візуалізації датасету flats.csv:",
     "images": [f("lab1_flats_bar.png"), f("lab1_flats_hist.png"), f("lab1_flats_scatter.png"), f("lab1_flats_box.png")]},
    {"type": "section", "text": "Відповіді на контрольні запитання:"},
    *[{"type": "text", "text": q} for q in QA],
    {"type": "section", "text": "Описова статистика та аналіз flats.csv:", "page_break_before": True},
    *[{"type": "text", "text": t} for t in FLATS_ANALYSIS],

    {"type": "task", "number": 2, "page_break_before": True,
     "text": "Завантажити, очистити та проаналізувати датасет Penguin Species; побудувати 4 обов'язкові діаграми: графік розсіювання, кругову діаграму, гістограму/KDE та коробчату діаграму (boxplot)."},
    {"type": "result", "label": "Результати візуалізації датасету Penguin Species:",
     "images": [f("lab1_peng_scatter.png"), f("lab1_peng_pie.png"),
                f("lab1_peng_hist.png"), f("lab1_peng_box.png")]},
    {"type": "section", "text": "Опис датасету та очищення:", "page_break_before": True},
    *[{"type": "text", "text": t} for t in PENG_DESC],
    {"type": "section", "text": "Аналіз діаграм:"},
    *[{"type": "text", "text": t} for t in PENG_ANALYSIS],
]

REPORT = {"meta": META, "body": BODY, "conclusion": CONCLUSION}


# ---------------------------------------------------------------------------
# Renderer (house style + inline bold + section labels)
# ---------------------------------------------------------------------------
def _set_font(run, family, size, bold=False):
    run.font.name = family
    run.font.size = size
    run.bold = bold
    rPr = run._r.get_or_add_rPr()
    rFonts = rPr.find(qn("w:rFonts"))
    if rFonts is None:
        rFonts = rPr.makeelement(qn("w:rFonts"), {})
        rPr.append(rFonts)
    rFonts.set(qn("w:ascii"), family)
    rFonts.set(qn("w:hAnsi"), family)
    rFonts.set(qn("w:cs"), family)


def _para(doc, text="", *, align=None, font=BODY_FONT, size=BODY_SIZE,
          bold=False, line_spacing=1.15, keep_with_next=False, keep_together=True):
    p = doc.add_paragraph()
    pf = p.paragraph_format
    if align is not None:
        p.alignment = align
    pf.line_spacing = line_spacing
    pf.space_after = Pt(0)
    if keep_together:
        pf.keep_together = True
    if keep_with_next:
        pf.keep_with_next = True
    if text:
        _set_font(p.add_run(text), font, size, bold)
    return p


LINK_RE = re.compile(r"\[([^\]]+)\]\((https?://[^)\s]+)\)")


def _add_bold(p, text, font=BODY_FONT, size=BODY_SIZE):
    """Add text with **bold** spans as separate runs."""
    for seg in re.split(r"(\*\*.+?\*\*)", text):
        if not seg:
            continue
        if seg.startswith("**") and seg.endswith("**") and len(seg) > 4:
            _set_font(p.add_run(seg[2:-2]), font, size, bold=True)
        else:
            _set_font(p.add_run(seg), font, size, bold=False)


def _add_hyperlink(p, url, text, font=BODY_FONT, size=BODY_SIZE):
    part = p.part
    r_id = part.relate_to(url, RT.HYPERLINK, is_external=True)
    hyperlink = OxmlElement("w:hyperlink")
    hyperlink.set(qn("r:id"), r_id)
    run = OxmlElement("w:r")
    rPr = OxmlElement("w:rPr")
    rFonts = OxmlElement("w:rFonts")
    for attr in ("w:ascii", "w:hAnsi", "w:cs"):
        rFonts.set(qn(attr), font)
    color = OxmlElement("w:color")
    color.set(qn("w:val"), "0563C1")
    u = OxmlElement("w:u")
    u.set(qn("w:val"), "single")
    sz = OxmlElement("w:sz")
    sz.set(qn("w:val"), str(int(size.pt * 2)))
    for el in (rFonts, color, u, sz):
        rPr.append(el)
    run.append(rPr)
    t = OxmlElement("w:t")
    t.set(qn("xml:space"), "preserve")
    t.text = text
    run.append(t)
    hyperlink.append(run)
    p._p.append(hyperlink)


def _add_rich(p, text, font=BODY_FONT, size=BODY_SIZE):
    """Add text with **bold** spans and [text](url) hyperlinks."""
    pos = 0
    for m in LINK_RE.finditer(text):
        if m.start() > pos:
            _add_bold(p, text[pos:m.start()], font, size)
        _add_hyperlink(p, m.group(2), m.group(1), font, size)
        pos = m.end()
    if pos < len(text):
        _add_bold(p, text[pos:], font, size)


def _add_image(doc, path):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(0)
    src = Path(path)
    if src.is_file():
        p.add_run().add_picture(str(src), width=FIG_WIDTH)
    else:
        _set_font(p.add_run(f"[TODO: missing image {path}]"), BODY_FONT, BODY_SIZE)
        print(f"  warning: image not found: {path}")


def _genitive(name):
    parts = name.split()
    if len(parts) < 2:
        return name
    sur, first = parts[0], parts[1]
    if sur.endswith(("й", "ї")):
        sur = sur[:-1] + "я"
    elif not sur.endswith(("а", "я")):
        sur += "а"
    if first.endswith(("й", "ї")):
        first = first[:-1] + "я"
    elif not first.endswith(("а", "я")):
        first += "а"
    return f"{sur} {first}"


def _title_block(doc, meta):
    c = WD_ALIGN_PARAGRAPH.CENTER
    lines = [
        meta.get("title_word", "ЗВІТ"),
        f"про виконання лабораторної роботи №{meta['lab_number']}",
        f"«{meta['topic']}»",
        "з дисципліни",
        f"«{meta['course_name']}»",
        f"Студента групи {meta['group']}",
        _genitive(meta["author"]),
    ]
    for i, line in enumerate(lines):
        _para(doc, line, align=c, bold=(i == len(lines) - 1))


def build_docx(data, out_docx):
    doc = Document()
    sec = doc.sections[0]
    sec.page_width, sec.page_height = Cm(21.0), Cm(29.7)
    sec.left_margin = sec.right_margin = Cm(MARGIN_CM)
    sec.top_margin = sec.bottom_margin = Cm(MARGIN_CM)
    sec.header_distance = sec.footer_distance = Cm(1.25)
    normal = doc.styles["Normal"]
    normal.font.name = BODY_FONT
    normal.font.size = BODY_SIZE
    normal.paragraph_format.line_spacing = 1.15
    normal.paragraph_format.space_after = Pt(0)

    _title_block(doc, data["meta"])

    if data["meta"].get("objective"):
        p = _para(doc, align=WD_ALIGN_PARAGRAPH.JUSTIFY, keep_with_next=True)
        _set_font(p.add_run("Мета роботи: "), BODY_FONT, BODY_SIZE, bold=True)
        _add_rich(p, data["meta"]["objective"])

    J, L = WD_ALIGN_PARAGRAPH.JUSTIFY, WD_ALIGN_PARAGRAPH.LEFT
    list_counter = 0
    for block in data["body"]:
        if block.get("page_break_before"):
            doc.add_page_break()
        btype = block["type"]
        if btype == "text":
            for line in block["text"].split("\n"):
                line = line.strip()
                if not line:
                    continue
                kwn = any(line.startswith(prefix) for prefix in ("**1.", "**2.", "**3.", "**4."))
                p = _para(doc, align=J, keep_with_next=kwn)
                _add_rich(p, line)
        elif btype == "task":
            list_counter = 0
            p = _para(doc, align=J, keep_with_next=True)
            _set_font(p.add_run(f"Завдання {block['number']}: "), BODY_FONT, BODY_SIZE, bold=True)
            _add_rich(p, block.get("text", ""))
        elif btype == "section":
            _para(doc, block["text"], align=J, bold=True, keep_with_next=True)
        elif btype == "code":
            list_counter += 1
            _para(doc, f"{list_counter}. {block['label']}", align=L, bold=True, keep_with_next=True)
            for line in block["text"].splitlines():
                _para(doc, line, align=L, font=CODE_FONT, size=CODE_SIZE,
                      line_spacing=Pt(9.5))
        elif btype == "result":
            lbl = f"{list_counter + 1}. {block['label']}" if list_counter > 0 else block["label"]
            _para(doc, lbl, align=L, bold=True, keep_with_next=True)
            for img in block["images"]:
                _add_image(doc, img)
        else:
            raise ValueError(f"unknown block type: {btype!r}")

    p = _para(doc, align=J)
    _set_font(p.add_run("Висновок: "), BODY_FONT, BODY_SIZE, bold=True)
    _add_rich(p, data["conclusion"])

    Path(out_docx).parent.mkdir(parents=True, exist_ok=True)
    doc.save(out_docx)
    print(f"wrote {out_docx}")


if __name__ == "__main__":
    out_json = os.path.join(BASE, "report.json")
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(REPORT, fh, ensure_ascii=False, indent=2)
    print("wrote", out_json)
    build_docx(REPORT, OUT_DOCX)
