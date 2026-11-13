"""
make_report.py
~~~~~~~~~~~~~~
Формирование отчёта report.docx по практике 3 на основе файла results.txt.
Отчёт создаётся в формате docx средствами стандартной библиотеки Python.
Группа: ЕТ-442
ФИО: Замятин Андрей Александрович
"""

import re
import zipfile

NS = ('xmlns:wpc="http://schemas.microsoft.com/office/word/2010/wordprocessingCanvas" '
      'xmlns:mc="http://schemas.openxmlformats.org/markup-compatibility/2006" '
      'xmlns:o="urn:schemas-microsoft-com:office:office" '
      'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" '
      'xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math" '
      'xmlns:wp14="http://schemas.microsoft.com/office/word/2010/wordprocessingDrawing" '
      'xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing" '
      'xmlns:w10="urn:schemas-microsoft-com:office:word" '
      'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" '
      'xmlns:w14="http://schemas.microsoft.com/office/word/2010/wordml" '
      'xmlns:wpg="http://schemas.microsoft.com/office/word/2010/wordprocessingGroup" '
      'xmlns:wpi="http://schemas.microsoft.com/office/word/2010/wordprocessingInk" '
      'xmlns:wne="http://schemas.microsoft.com/office/word/2006/wordml" '
      'xmlns:wps="http://schemas.microsoft.com/office/word/2010/wordprocessingShape" '
      'mc:Ignorable="w14 wp14"')


def esc(text):
    return (text.replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace('"', "&quot;"))


def para(text="", bold=False, align=None, size=28, font="Times New Roman",
         page_break_after=False):
    """Один абзац; size в полупунктах (28 = 14 pt)."""
    ppr = ""
    if align:
        ppr = '<w:pPr><w:jc w:val="%s"/></w:pPr>' % align
    if not text and not page_break_after:
        return '<w:p>%s</w:p>' % ppr
    run = ""
    if text:
        run = ('<w:r><w:rPr><w:rFonts w:ascii="%s" w:hAnsi="%s" w:cs="%s"/>%s'
               '<w:sz w:val="%d"/><w:szCs w:val="%d"/></w:rPr>'
               '<w:t xml:space="preserve">%s</w:t></w:r>'
               % (font, font, font, "<w:b/>" if bold else "",
                  size, size, esc(text)))
    if page_break_after:
        run += '<w:r><w:br w:type="page"/></w:r>'
    return '<w:p>%s%s</w:p>' % (ppr, run)


def cell(text, bold=False, size=24):
    return ('<w:tc><w:tcPr><w:tcW w:w="1800" w:type="dxa"/></w:tcPr>'
            + para(text, bold=bold, align="center", size=size) + '</w:tc>')


def table(header, rows):
    """Простая таблица с границами."""
    tbl = ['<w:tbl><w:tblPr><w:tblW w:w="9000" w:type="dxa"/>'
           '<w:jc w:val="center"/>'
           '<w:tblBorders>'
           '<w:top w:val="single" w:sz="4" w:space="0" w:color="000000"/>'
           '<w:left w:val="single" w:sz="4" w:space="0" w:color="000000"/>'
           '<w:bottom w:val="single" w:sz="4" w:space="0" w:color="000000"/>'
           '<w:right w:val="single" w:sz="4" w:space="0" w:color="000000"/>'
           '<w:insideH w:val="single" w:sz="4" w:space="0" w:color="000000"/>'
           '<w:insideV w:val="single" w:sz="4" w:space="0" w:color="000000"/>'
           '</w:tblBorders></w:tblPr>']
    tbl.append('<w:tr>' + "".join(cell(h, bold=True) for h in header) + '</w:tr>')
    for row in rows:
        tbl.append('<w:tr>' + "".join(cell(c) for c in row) + '</w:tr>')
    tbl.append('</w:tbl>')
    return "".join(tbl)


def parse_results(filename="results.txt"):
    """Читает results.txt и возвращает списки метрик по эпохам и сводку."""
    text = open(filename, encoding="utf-8").read()
    epochs = []
    cur = {}
    for line in text.splitlines():
        m = re.match(r"Epoch (\d+) training complete", line)
        if m:
            if cur:
                epochs.append(cur)
            cur = {"epoch": int(m.group(1))}
            continue
        m = re.match(r"--Cost on training data: ([0-9.]+)", line)
        if m:
            cur["train_cost"] = float(m.group(1))
            continue
        m = re.match(r"--Accuracy on training data: (\d+) / (\d+)", line)
        if m:
            cur["train_acc"] = int(m.group(1))
            continue
        m = re.match(r"--Cost on evaluation data: ([0-9.]+)", line)
        if m:
            cur["eval_cost"] = float(m.group(1))
            continue
        m = re.match(r"--Accuracy on evaluation data: (\d+) / (\d+)", line)
        if m:
            cur["eval_acc"] = int(m.group(1))
            continue
    if cur:
        epochs.append(cur)
    m = re.search(r"Проверка загруженной из файла сети на тестовой выборке: "
                  r"(\d+) / (\d+) \(([0-9.]+)%\)", text)
    test = (int(m.group(1)), int(m.group(2)), float(m.group(3))) if m else None
    m = re.search(r"Время обучения: ([0-9.]+) с \(([0-9.]+) мин\)", text)
    duration = (float(m.group(1)), float(m.group(2))) if m else None
    return epochs, test, duration


def build_document(epochs, test, duration):
    best = max(epochs, key=lambda e: e["eval_acc"])
    parts = []

    # --- Титульный лист ---
    parts.append(para("Министерство науки и высшего образования Российской Федерации",
                      align="center"))
    parts.append(para("Федеральное государственное автономное образовательное учреждение "
                      "высшего образования", align="center"))
    parts.append(para("«Южно-Уральский государственный университет "
                      "(национальный исследовательский университет)»", align="center"))
    parts.append(para("Кафедра системного программирования", align="center"))
    parts.append(para())
    parts.append(para("ОТЧЁТ", bold=True, align="center", size=32))
    parts.append(para("по практической работе № 3", align="center"))
    parts.append(para("по дисциплине «Глубокое обучение»", align="center"))
    parts.append(para("Тема: «Распознавание рукописных цифр нейронной сетью "
                      "с использованием стоимостной функции на основе "
                      "перекрестной энтропии»", align="center"))
    parts.append(para())
    parts.append(para("Выполнил: студент группы ЕТ-442", align="left"))
    parts.append(para("Замятин Андрей Александрович", align="left"))
    for _ in range(8):
        parts.append(para())
    parts.append(para("Челябинск, 2026", align="center", page_break_after=True))

    # --- 1. Цель работы ---
    parts.append(para("1. Цель работы", bold=True))
    parts.append(para("Цель работы — написать компьютерную программу на языке Python 3, "
                      "создающую и обучающую нейронную сеть для распознавания рукописных "
                      "цифр с использованием метода градиентного спуска и стоимостной "
                      "функции на основе перекрестной энтропии, оценить достигнутый "
                      "прогресс в обучении и сохранить обученную нейронную сеть в файл.",
                      align="both"))

    # --- 2. Краткие теоретические сведения ---
    parts.append(para("2. Краткие теоретические сведения", bold=True))
    parts.append(para("Программа network2.py является развитием программы network.py "
                      "из практических работ 1 и 2 и отличается от неё тремя основными "
                      "новшествами.", align="both"))
    parts.append(para("Во-первых, изменён способ инициализации весов: в методе "
                      "default_weight_initializer случайное значение веса делится на "
                      "квадратный корень из числа входных сигналов нейрона. В старой "
                      "версии (метод large_weight_initializer сохранён для сравнения) "
                      "веса выбирались из стандартного нормального распределения, "
                      "поэтому у нейрона со многими входами активационный потенциал "
                      "имел большое среднеквадратическое отклонение, сигмоида сразу "
                      "насыщалась, её производная становилась близкой к нулю и обучение "
                      "замедлялось. Деление на корень из числа входов делает "
                      "распределение весов более узким, сигмоида работает в крутой "
                      "области, где её производная велика, и сеть обучается быстрее.",
                      align="both"))
    parts.append(para("Во-вторых, вместо среднеквадратичной стоимостной функции "
                      "(класс QuadraticCost, как в предыдущей версии программы) "
                      "используется стоимостная функция на основе перекрестной "
                      "энтропии (класс CrossEntropyCost). При квадратичной функции "
                      "потерь градиент на выходном слое содержит множитель в виде "
                      "производной сигмоиды, из-за чего при насыщении нейрона "
                      "скорость обучения падает. У перекрестной энтропии этот "
                      "множитель сокращается: мера влияния нейрона выходного слоя "
                      "на ошибку равна просто разности фактического и ожидаемого "
                      "выходов, поэтому чем сильнее сеть ошибается, тем быстрее "
                      "она обучается.", align="both"))
    parts.append(para("В-третьих, в метод градиентного спуска добавлен механизм "
                      "L2-регуляризации с параметром сглаживания lmbda: при обновлении "
                      "веса дополнительно умножаются на коэффициент (1 - eta*lmbda/n), "
                      "что штрафует сеть за слишком большие веса и уменьшает "
                      "переобучение. Кроме того, добавлены методы save и load для "
                      "сохранения обученной сети в файл и её последующей загрузки, "
                      "а также возможность мониторинга значения стоимостной функции "
                      "и точности на обучающей и оценочной выборках после каждой "
                      "эпохи.", align="both"))

    # --- 3. Ход работы ---
    parts.append(para("3. Ход работы", bold=True))
    parts.append(para("В каталоге practical-3 создан модуль network2.py в соответствии "
                      "с методическими указаниями, подключён модуль mnist_loader и "
                      "инициализированы наборы данных:", align="both"))
    parts.append(para(">>> import mnist_loader", align="left", font="Consolas"))
    parts.append(para(">>> training_data, validation_data, test_data = "
                      "mnist_loader.load_data_wrapper()", align="left", font="Consolas"))
    parts.append(para(">>> import network2", align="left", font="Consolas"))
    parts.append(para("Создана нейронная сеть для распознавания рукописных цифр "
                      "со стоимостной функцией на основе перекрестной энтропии:",
                      align="both"))
    parts.append(para(">>> net = network2.Network([784, 30, 10], "
                      "cost=network2.CrossEntropyCost)", align="left", font="Consolas"))
    parts.append(para("Запущена процедура обучения, включающая 30 эпох, со всеми "
                      "включёнными флагами мониторинга:", align="both"))
    parts.append(para(">>> net.SGD(training_data, 30, 10, 0.5, lmbda=5.0, "
                      "evaluation_data=validation_data, "
                      "monitor_evaluation_accuracy=True, "
                      "monitor_evaluation_cost=True, "
                      "monitor_training_accuracy=True, "
                      "monitor_training_cost=True)", align="left", font="Consolas"))
    parts.append(para("Обучающая выборка — 50000 изображений, оценочная — 10000, "
                      "тестовая — 10000. Размер мини-пакета — 10, скорость обучения "
                      "eta = 0.5, параметр регуляризации lmbda = 5.0.", align="both"))

    # --- 4. Результаты обучения ---
    parts.append(para("4. Результаты обучения", bold=True))
    parts.append(para("В таблице приведены значения стоимостной функции (перекрестной "
                      "энтропии) и точности (числа верно распознанных изображений) на "
                      "обучающей и оценочной выборках после каждой эпохи обучения.",
                      align="both"))
    header = ["Эпоха", "Cost (обуч.)", "Точность (обуч.)",
              "Cost (оцен.)", "Точность (оцен.)"]
    rows = [[str(e["epoch"]), "%.4f" % e["train_cost"],
             "%d / 50000" % e["train_acc"], "%.4f" % e["eval_cost"],
             "%d / 10000" % e["eval_acc"]] for e in epochs]
    parts.append(table(header, rows))
    parts.append(para())

    # --- 5. Ответ на вопрос задания ---
    parts.append(para("5. Ответ на вопрос задания", bold=True))
    parts.append(para("Какого прогресса в обучении удалось достигнуть?", bold=True,
                      align="both"))
    cost_dir = ("снизилось" if epochs[-1]["eval_cost"] < epochs[0]["eval_cost"]
                else "увеличилось")
    parts.append(para("За 30 эпох обучения точность распознавания на оценочной выборке "
                      "выросла с %d из 10000 на эпохе 0 до %d из 10000 на эпохе 29 "
                      "(%.2f%%). Лучший результат — %d из 10000 (%.2f%%) — достигнут "
                      "на эпохе %d. Значение стоимостной функции на оценочной выборке "
                      "%s с %.4f до %.4f (рост значения при одновременном росте "
                      "точности объясняется тем, что сеть всё увереннее ошибается "
                      "на немногих сложных примерах — начальная стадия переобучения). "
                      "На обучающей выборке точность выросла "
                      "с %d до %d из 50000."
                      % (epochs[0]["eval_acc"], epochs[-1]["eval_acc"],
                         epochs[-1]["eval_acc"] / 100.0,
                         best["eval_acc"], best["eval_acc"] / 100.0, best["epoch"],
                         cost_dir, epochs[0]["eval_cost"], epochs[-1]["eval_cost"],
                         epochs[0]["train_acc"], epochs[-1]["train_acc"]),
                      align="both"))
    if test:
        parts.append(para("Обученная сеть сохранена в файл trained_network.json "
                          "методом save. Загруженная из файла сеть (метод load) "
                          "проверена на тестовой выборке и показала точность "
                          "%d из %d (%.2f%%), что подтверждает корректность "
                          "сохранения и загрузки." % (test[0], test[1], test[2]),
                          align="both"))
    if duration:
        parts.append(para("Обучение заняло %.2f минуты." % duration[1], align="both"))
    parts.append(para("Для сравнения: в практической работе 2 та же сеть [784, 30, 10] "
                      "с квадратичной функцией потерь и старой инициализацией весов при "
                      "скорости обучения 3.0 показывала на тестовой выборке 90.75%% на "
                      "нулевой эпохе и около 95.2%% к концу 30 эпох. Использование "
                      "перекрестной энтропии и улучшенной инициализации весов ускорило "
                      "обучение: уже на нулевой эпохе достигнуто %.2f%%, а начиная "
                      "с эпохи 5 точность на оценочной выборке превышает 96%%. "
                      "Заметно также, что при росте точности значение стоимостной "
                      "функции на оценочной выборке к концу обучения несколько "
                      "увеличилось — это характерный признак начинающегося "
                      "переобучения, частично сдерживаемый L2-регуляризацией."
                      % (epochs[0]["eval_acc"] / 100.0,), align="both"))

    # --- 6. Вывод ---
    parts.append(para("6. Вывод", bold=True))
    parts.append(para("В ходе работы создана и обучена нейронная сеть для распознавания "
                      "рукописных цифр со стоимостной функцией на основе перекрестной "
                      "энтропии, L2-регуляризацией и улучшенной инициализацией весов. "
                      "Достигнута точность около %.2f%% на оценочной выборке. "
                      "Обученная сеть сохранена в файл trained_network.json, "
                      "который прилагается к отчёту." % (best["eval_acc"] / 100.0),
                      align="both"))

    body = "".join(parts)
    return ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<w:document %s><w:body>%s'
            '<w:sectPr><w:pgSz w:w="11906" w:h="16838"/>'
            '<w:pgMar w:top="1134" w:right="850" w:bottom="1134" w:left="1701" '
            'w:header="708" w:footer="708" w:gutter="0"/></w:sectPr>'
            '</w:body></w:document>') % (NS, body)


CONTENT_TYPES = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                 '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
                 '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
                 '<Default Extension="xml" ContentType="application/xml"/>'
                 '<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
                 '</Types>')

RELS = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>'
        '</Relationships>')


def main():
    epochs, test, duration = parse_results("results.txt")
    if not epochs:
        raise SystemExit("results.txt не содержит данных об эпохах — "
                         "сначала выполните run_training.py")
    document = build_document(epochs, test, duration)
    with zipfile.ZipFile("report.docx", "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", CONTENT_TYPES)
        z.writestr("_rels/.rels", RELS)
        z.writestr("word/document.xml", document)
    print("Отчёт сохранён в report.docx (эпох: %d)" % len(epochs))


if __name__ == "__main__":
    main()
