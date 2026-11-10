"""Baseline findings, written so the results can be explained without the scripts."""

from fpdf import FPDF

FONT = "/System/Library/Fonts/Supplemental"
OUT = "/Users/aryan/Desktop/VRPS- Research Associate/VRPS_Baseline_Findings.pdf"
NAVY = (22, 54, 84)
INK = (35, 44, 52)
MUTED = (90, 100, 110)
RULE = (186, 198, 210)
BAND = (240, 244, 248)


class Doc(FPDF):
    def header(self):
        if self.page_no() == 1:
            return
        saved = self.get_y()
        self.set_xy(self.l_margin, 8)
        self.set_font("Arial", "", 9)
        self.set_text_color(*MUTED)
        self.cell(0, 5, "VRPS    Baseline findings    31 October 2026", align="L")
        self.set_draw_color(*RULE)
        self.line(self.l_margin, 15, self.w - self.r_margin, 15)
        self.set_y(saved)

    def footer(self):
        self.set_y(-12)
        self.set_font("Arial", "", 9)
        self.set_text_color(120, 125, 130)
        self.cell(0, 8, str(self.page_no()), align="C")


def h1(pdf, text):
    if pdf.get_y() > 250:
        pdf.add_page()
    pdf.ln(1)
    pdf.set_font("Arial", "B", 13)
    pdf.set_text_color(*NAVY)
    pdf.multi_cell(0, 7, text)
    pdf.ln(1)


def h2(pdf, text):
    if pdf.get_y() > 235:
        pdf.add_page()
    pdf.ln(2.5)
    pdf.set_font("Arial", "B", 12)
    pdf.set_text_color(*NAVY)
    pdf.multi_cell(0, 6.2, text)
    pdf.ln(0.6)


def body(pdf, text):
    pdf.set_font("Arial", "", 10.5)
    pdf.set_text_color(*INK)
    pdf.multi_cell(0, 5.2, text)
    pdf.ln(1.1)


def lead(pdf, label, text):
    pdf.set_text_color(*INK)
    pdf.set_font("Arial", "B", 10.5)
    pdf.write(5.2, label)
    pdf.set_font("Arial", "", 10.5)
    pdf.write(5.2, "  " + text)
    pdf.ln(6.2)


def table(pdf, headers, rows, widths):
    pdf.set_font("Arial", "B", 9)
    pdf.set_fill_color(*BAND)
    pdf.set_text_color(*NAVY)
    pdf.set_draw_color(*RULE)
    line_h = 6.6
    for header, width in zip(headers, widths):
        pdf.cell(width, line_h, header, border=1, fill=True)
    pdf.ln()
    pdf.set_font("Arial", "", 9)
    pdf.set_text_color(*INK)
    for i, row in enumerate(rows):
        pdf.set_fill_color(248, 250, 252)
        fill = i % 2 == 1
        for value, width in zip(row, widths):
            pdf.cell(width, line_h, value, border=1, fill=fill)
        pdf.ln()
    pdf.ln(1.6)


def main():
    pdf = Doc(format="A4")
    pdf.set_auto_page_break(auto=True, margin=16)
    pdf.set_margins(16, 22, 16)
    for style, filename in (("", "Arial.ttf"), ("B", "Arial Bold.ttf"), ("I", "Arial Italic.ttf")):
        pdf.add_font("Arial", style, f"{FONT}/{filename}")

    pdf.add_page()
    pdf.set_fill_color(*NAVY)
    pdf.rect(0, 0, 210, 32, style="F")
    pdf.set_y(10)
    pdf.set_x(16)
    pdf.set_font("Arial", "", 10)
    pdf.set_text_color(190, 210, 225)
    pdf.cell(0, 5, "VISVESVARAYA RESEARCH PROMOTION SCHEME")
    pdf.ln(6)
    pdf.set_x(16)
    pdf.set_font("Arial", "B", 16)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(0, 8, "Baseline findings")
    pdf.set_y(38)

    body(
        pdf,
        "Updated 31 October 2026. The scores are from the 15 October runs, seed 42. "
        "Six separate scripts, one question each. They are not yet one decision system.",
    )

    h1(pdf, "Where the code is")
    body(
        pdf,
        "Folder: development. Every module has a README.md, train_baselines.py, download_data.py, "
        "requirements.txt, a data folder, and a results folder. The README is the long note. "
        "train_baselines.py is the script to walk through. results/model_comparison.csv is the score table. "
        "The public files are already downloaded. Run a module from its own folder with python train_baselines.py.",
    )

    h1(pdf, "How to read a score")
    lead(
        pdf,
        "Majority class.",
        "Always guess the most common label. On the soil file that guess is right half the time, so accuracy 0.50 is the floor. A model we keep has to beat this.",
    )
    lead(
        pdf,
        "Macro F1.",
        "A single score that gives every class the same weight. Accuracy can look high while the rare class is mostly wrong. Quote macro F1 for soil, leaves, fertilizer, and crop.",
    )
    lead(
        pdf,
        "MAE.",
        "Average size of the miss, in the same units as the target. Quote MAE for moisture and yield.",
    )
    lead(
        pdf,
        "R\u00b2.",
        "1 is a perfect fit. 0 is no better than guessing the average. Below 0 is worse than guessing the average.",
    )
    lead(
        pdf,
        "Persistence.",
        "For the moisture sensor: predict that the reading will not change. This is the number the other models have to beat.",
    )
    lead(
        pdf,
        "Crop mean.",
        "For yield: predict that crop's usual yield from the training years. A model that only recognises coconut will beat a flat average and still lose to this.",
    )

    h1(pdf, "What to quote")
    table(
        pdf,
        ["Module", "Quote this", "Next step"],
        [
            ["1  Soil", "Forest macro F1  0.864", "More unique Highly Fertile rows"],
            ["2  Moisture", "Persistence MAE  0.00073", "Add rainfall. Keep this baseline"],
            ["3  Leaves", "16 test photos, 3 diseases", "Keep it. Do not quote 1.00"],
            ["4  Fertilizer", "Forest F1  0.967 \u00b1 0.065", "Keep it. Do not tune"],
            ["5  Yield", "Forest MAE  1.65, no coconut", "Score coconut and Bajra apart"],
            ["6  Crop", "Forest macro F1  0.993", "Need a file that has millet"],
        ],
        [36, 68, 74],
    )

    h2(pdf, "1. Soil fertility")
    lead(pdf, "Folder.", "module_01_soil_nutrient")
    lead(
        pdf,
        "Question.",
        "Which fertility class is already written on this soil row: Less Fertile, Fertile, or Highly Fertile.",
    )
    lead(
        pdf,
        "Data.",
        "Public soil table. 1,288 rows in the file, 880 after removing exact copies. The copies were all Highly Fertile. Test set is 176 rows.",
    )
    lead(
        pdf,
        "Result.",
        "Majority class macro F1 0.222. A short decision tree 0.853. Random forest 0.864. Logistic regression 0.666. SVM 0.708.",
    )
    lead(
        pdf,
        "Say this.",
        "Keep the forest. The short tree is close behind, so changing the algorithm will not fix the rare class. Nitrogen and phosphorus do most of the separating. Highly Fertile scores 0.71 because that class has 8 test rows.",
    )
    lead(
        pdf,
        "Leave out.",
        "Macro F1 0.961 on the file with the copies left in. 90 of those test rows were copies of training rows.",
    )
    lead(pdf, "Next.", "More unique Highly Fertile samples. Do not retune the forest on 39 rare rows.")

    h2(pdf, "2. Soil moisture")
    lead(pdf, "Folder.", "module_02_irrigation_soil_moisture")
    lead(
        pdf,
        "Question.",
        "What will the water-content sensor read in 4 hours. The file has no irrigate / do-not-irrigate label.",
    )
    lead(
        pdf,
        "Data.",
        "15-minute sensors from an olive field in Gallipoli, Italy. The test period is the last part of the calendar, so the model cannot see the future during training.",
    )
    lead(
        pdf,
        "Result, 4 hours, port 1.",
        "Persistence MAE 0.00073. Linear regression 0.00123. Random forest 0.00221. Yesterday's reading 0.00337.",
    )
    lead(
        pdf,
        "Say this.",
        "Over 4 hours the reading barely moves, so predicting no change wins. The forest spends almost all of its attention on the current reading and still misses more than persistence. At 24 hours the forest is worse than no change (R\u00b2 \u22120.26). The same pattern is stronger on port 2.",
    )
    lead(
        pdf,
        "Leave out.",
        "The linear SVM. It did not finish fitting, so its small edges are not a result.",
    )
    lead(
        pdf,
        "Next.",
        "Add rainfall and irrigation from the same download, and still compare with persistence. Another tree model will not change this.",
    )

    h2(pdf, "3. Pearl millet leaves")
    lead(pdf, "Folder.", "module_03_pearl_millet_disease")
    lead(
        pdf,
        "Question.",
        "Is this leaf healthy, downy mildew (Sclerospora), or leaf blight (Exserohilum).",
    )
    lead(
        pdf,
        "Data.",
        "80 photos: 41 healthy, 30 Sclerospora, 9 Exserohilum. Each photo is turned into 30 colour numbers. 64 photos to train, 16 to test. Smut and Pyricularia are omitted here because those downloads are about 230 MB and 370 MB.",
    )
    lead(
        pdf,
        "Result.",
        "Majority class macro F1 0.222. Decision tree 1.000. Logistic regression and linear SVM 0.869. Random forest 0.863.",
    )
    lead(
        pdf,
        "Say this.",
        "The tree's perfect score is 16 photos, including 2 blight photos. That is too small to trust. The forest misses one of those two blight photos. This is a colour baseline on three diseases, from research plots in Burkina Faso.",
    )
    lead(pdf, "Next.", "Keep this script as the light baseline. Add the two large disease sets only when there is disk space. A neural network comes after that, not before.")

    h2(pdf, "4. Fertilizer name")
    lead(pdf, "Folder.", "module_04_fertilizer_recommendation")
    lead(
        pdf,
        "Question.",
        "Which fertilizer name in the table matches this soil and crop row.",
    )
    lead(
        pdf,
        "Data.",
        "The raw file has 198 rows because 99 recommendations were pasted twice, and the grade 10-26-26 was saved as a date. The script fixes both before scoring. Seven fertilizer names. Two of them have only 7 rows. The score is the average of five reshuffles of those 99 rows.",
    )
    lead(
        pdf,
        "Result.",
        "Majority class macro F1 0.052. Tree 0.791. SVM 0.885. Random forest and gradient boosting both 0.967, and that number moves by about 0.065 from fold to fold.",
    )
    lead(
        pdf,
        "Say this.",
        "The forest and boosting tie. Keep the forest. Phosphorus, nitrogen, and potassium do most of the work, so on this table the fertilizer name is mostly a lookup from those three nutrients. The one miss is the grade 17-17-17. An accuracy of 0.99 and a swing of about 0.07 are both true, because 99 rows is a small file.",
    )
    lead(pdf, "Next.", "Keep the forest. Do not keep trying settings to push 0.967 higher. The next useful file is recommendations that were not built from the same nitrogen, phosphorus, and potassium rules.")

    h2(pdf, "5. Crop yield")
    lead(pdf, "Folder.", "module_05_crop_yield")
    lead(
        pdf,
        "Question.",
        "What yield was recorded for a later year. Training years are 1997 to 2016. Test years are 2017 to 2020 (3,249 rows).",
    )
    lead(
        pdf,
        "Data.",
        "Indian state statistics. Bajra is the pearl-millet name in this file. Production is left out, because yield is essentially production divided by area. Fertilizer and pesticide in this file are a year rate times area. Adding them changes the forest MAE from 12.67 to 12.15, so they stay out.",
    )
    table(
        pdf,
        ["Model", "All crops", "No coconut", "Bajra (85 rows)"],
        [
            ["Usual yield for that crop", "MAE 22.84", "MAE 2.48", "MAE 1.65"],
            ["Gradient boosting", "MAE 12.39", "MAE 2.55", "MAE 1.07"],
            ["Random forest", "MAE 12.67", "MAE 1.65", "MAE 0.81"],
        ],
        [58, 40, 40, 40],
    )
    lead(
        pdf,
        "Say this.",
        "Boosting wins the all-crop number because coconut yields sit between about 13,000 and 21,105, while Bajra sits near 1. Crop identity alone already explains most of that (R\u00b2 0.864). Once coconut is taken out of the score, only the forest beats the usual yield for that crop. On Bajra, the forest cuts the average miss from 1.65 to 0.81. That is 85 rows of state statistics, not a field trial.",
    )
    lead(
        pdf,
        "Also say.",
        "On bajra, small millets, jowar, and ragi together (327 rows), the forest's typical miss stays close to the crop average, and R\u00b2 falls to \u22121.46. A few rows pull that R\u00b2. Quote the median miss there, and keep the rows.",
    )
    lead(pdf, "Next.", "Carry the forest. Report coconut, the other crops, and Bajra as three numbers. Do not drop the large yields to make the error look smaller.")

    h2(pdf, "6. Crop recommendation")
    lead(pdf, "Folder.", "module_06_crop_recommendation")
    lead(
        pdf,
        "Question.",
        "Which of 22 crop labels matches this soil and climate row.",
    )
    lead(
        pdf,
        "Data.",
        "2,200 rows, 100 per crop. The source says extra rows were generated to fill the table. There is no millet class. Test set is 440 rows.",
    )
    lead(
        pdf,
        "Result.",
        "Majority class accuracy 0.045. Decision tree macro F1 0.852. Logistic regression 0.972. SVM 0.984. Random forest 0.993.",
    )
    lead(
        pdf,
        "Say this.",
        "The classes in this file are easy to separate. The misses are single rows. Rainfall and humidity matter more than N, P, K, or pH. pH is the smallest input. Moving pH from 4.5 to 8.5 on a typical rice, maize, cotton, or chickpea row changes the prediction zero times.",
    )
    lead(
        pdf,
        "Leave this open.",
        "An earlier project note said that lowering pH produced a poor recommendation. This forest, on these four typical rows, does not show that. The note may have come from another script. This result does not close it.",
    )
    lead(pdf, "Next.", "Keep the forest. Do not tune it toward 0.995. The useful next file is one that contains the millet crops.")

    h1(pdf, "What we have, in one pass")
    body(
        pdf,
        "Each module runs, writes a comparison, and has a README. Soil, fertilizer, and crop are public tables, and the high scores there describe those tables. Moisture is an Italian olive field, and the honest 4-hour forecast is no change. Leaves are 80 photos of three diseases. Yield is state statistics. For this project, quote the Bajra number, and quote coconut separately so the large coconut yields cannot carry the score.",
    )
    body(
        pdf,
        "None of these scripts reads the project sensors, and none of them is a combined recommendation yet. Do the next step named on each module. Deal with the rare soil class, the 16 leaf photos, and the coconut yields before spending time on the fertilizer and crop models, which are already above 0.95 on easy tables.",
    )

    pdf.output(OUT)
    print(OUT, "pages", pdf.page)


if __name__ == "__main__":
    main()
