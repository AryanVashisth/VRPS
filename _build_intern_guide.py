"""Build the intern PDF. This file is a document generator, not a module solution."""

from fpdf import FPDF
from fpdf.fonts import TextStyle

FONT = "/System/Library/Fonts/Supplemental"
OUT = "/Users/aryan/Desktop/VRPS- Research Associate/VRPS_Intern_Guide_ML_Modules.pdf"


class Guide(FPDF):
    def header(self):
        if self.page_no() == 1:
            return
        # add_page places the cursor at the top margin, then calls header.
        # Restore that cursor so body text does not sit on the running head.
        saved_y = self.get_y()
        self.set_xy(self.l_margin, 8)
        self.set_font("Arial", "", 9)
        self.set_text_color(90, 100, 110)
        self.cell(0, 5, "VRPS    Machine learning modules    intern guide", align="L")
        self.set_draw_color(186, 198, 210)
        self.line(self.l_margin, 15, self.w - self.r_margin, 15)
        self.set_y(saved_y)

    def footer(self):
        if self.page_no() == 1:
            return
        self.set_y(-12)
        self.set_font("Arial", "", 9)
        self.set_text_color(120, 125, 130)
        self.cell(0, 8, str(self.page_no()), align="C")


def cover(pdf: Guide) -> None:
    pdf.add_page()
    pdf.set_fill_color(22, 54, 84)
    pdf.rect(0, 0, 210, 78, style="F")
    pdf.set_y(22)
    pdf.set_x(18)
    pdf.set_font("Arial", "", 12)
    pdf.set_text_color(190, 210, 225)
    pdf.cell(0, 8, "VISVESVARAYA RESEARCH PROMOTION SCHEME")
    pdf.ln(12)
    pdf.set_x(18)
    pdf.set_font("Arial", "B", 26)
    pdf.set_text_color(255, 255, 255)
    pdf.multi_cell(174, 12, "Machine learning for the six modules")
    pdf.set_y(92)
    pdf.set_x(18)
    pdf.set_font("Arial", "", 13)
    pdf.set_text_color(22, 54, 84)
    pdf.multi_cell(174, 7, "Enhancing Millet Crop Production Using AI and IoT")
    pdf.ln(4)
    pdf.set_x(18)
    pdf.set_font("Arial", "", 12)
    pdf.set_text_color(50, 60, 70)
    pdf.multi_cell(
        174,
        7,
        "A guide for interns. It explains the algorithms used in each module, and how to implement and run that module yourself.",
    )
    pdf.ln(8)
    pdf.set_fill_color(240, 244, 248)
    pdf.set_x(18)
    pdf.set_font("Arial", "", 11)
    pdf.set_text_color(30, 40, 50)
    pdf.multi_cell(
        174,
        6.5,
        "Write the code yourself from this guide. The reference implementation will be shared after you have attempted the modules. A number you did not compute is not your result.",
        fill=True,
    )
    pdf.ln(8)
    pdf.set_x(18)
    pdf.set_font("Arial", "B", 12)
    pdf.set_text_color(22, 54, 84)
    pdf.cell(0, 8, "Contents")
    pdf.ln(8)
    pdf.set_font("Arial", "", 11)
    pdf.set_text_color(40, 50, 60)
    for line in [
        "1.  How to use this guide",
        "2.  The six modules",
        "3.  Ideas every module uses",
        "4.  The algorithms",
        "5.  How to build and run a module",
        "6.  Module by module",
        "7.  What to hand in",
        "8.  Checklist before you trust a score",
        "9.  Glossary",
    ]:
        pdf.set_x(18)
        pdf.cell(0, 7, line)
        pdf.ln(7)
    pdf.ln(6)
    pdf.set_x(18)
    pdf.set_font("Arial", "", 11)
    pdf.set_text_color(80, 90, 100)
    pdf.cell(0, 7, "October 2026")


HTML = r"""
<h1>1. How to use this guide</h1>
<p>There are six prediction tasks. Each one is its own small program. They are not combined into one model at this stage. Later, a decision step will read their outputs. That step is not part of this work.</p>
<p>For each module you will choose a public dataset, define the target, train a few standard algorithms on the same split, and write down what the numbers mean. The algorithms below are candidates. None of them has been declared the winner in advance.</p>
<p>Use Python with pandas, scikit-learn, and matplotlib. The leaf module also uses Pillow to open images. Do not install PyTorch for the first version. A neural network is a later experiment, and the library is large.</p>

<h1>2. The six modules</h1>
<table width="100%">
<thead>
<tr>
<th width="18%">Module</th>
<th width="32%">Question</th>
<th width="18%">Answer</th>
<th width="32%">Algorithms to implement</th>
</tr>
</thead>
<tbody>
<tr><td>1. Soil</td><td>Which fertility class is this sample?</td><td>A class</td><td>Tree, logistic regression, forest, SVM</td></tr>
<tr><td>2. Moisture</td><td>What will the sensor read later?</td><td>A number</td><td>Persistence, linear regression, forest, SVR</td></tr>
<tr><td>3. Disease</td><td>Is this pearl-millet leaf diseased?</td><td>A class</td><td>Tree, logistic regression, forest, linear SVM</td></tr>
<tr><td>4. Fertilizer</td><td>Which fertilizer name fits?</td><td>A class</td><td>Tree, forest, SVM, gradient boosting</td></tr>
<tr><td>5. Yield</td><td>What yield is recorded next year?</td><td>A number</td><td>Mean baselines, linear regression, tree, forest, boosting</td></tr>
<tr><td>6. Crop</td><td>Which crop label matches?</td><td>A class</td><td>Tree, logistic regression, forest, SVM</td></tr>
</tbody>
</table>
<p>Classification means the answer is a label, such as Fertile or Urea. Regression means the answer is a number, such as a yield or a water-content reading.</p>

<h1>3. Ideas every module uses</h1>
<h2>3.1 Features and target</h2>
<p>A feature is an input column: nitrogen, rainfall, the current sensor reading, a colour from a leaf. The target is the column you are trying to predict. You never put the target into the feature list. If another column is just the target written in a different form, leave that column out as well. That is target leakage, and the score then looks better than the model is.</p>
<h2>3.2 Train and test</h2>
<p>The model learns from the training rows. You measure it on rows it did not learn from. For a table of independent samples, use a stratified split when the target is a class: each class keeps the same share in train and test. Use random_state=42 so a second run gives the same split.</p>
<p>Do not shuffle a time series. Moisture and yield are ordered in time. Train on the earlier period and test on the later period. A random split lets the model see tomorrow while predicting today.</p>
<p>Fit every scaler, imputer, and encoder on the training rows only. The clean way is a scikit-learn Pipeline. You call fit on the pipeline. It learns the scaler from the training rows, then applies that same scaler to the test rows.</p>
<h2>3.3 A simple baseline</h2>
<p>Every comparison needs a dumb answer. For classes, predict the most common training class every time. For a number, predict the training average. For soil moisture, predict that the reading will stay where it is now. If a clever model does not beat that rule, the clever model is not useful yet.</p>
<h2>3.4 Metrics</h2>
<p>Accuracy is the share of labels that are correct. It hides a rare class. Also report macro precision, macro recall, and macro F1. Macro means you score each class, then average the classes, so a small class counts as much as a large one. Save the confusion matrix. It shows which classes are swapped.</p>
<p>For a number, report MAE, RMSE, and R-squared. MAE is the average size of the error. RMSE punishes a few very large errors. R-squared compares you with predicting the average. A negative R-squared means you are worse than predicting the average. Also look at the median absolute error when a few extreme values dominate RMSE.</p>
<h2>3.5 What you must not claim</h2>
<p>A high score on a public table is a statement about that table. It is not a statement about a millet field, and it is not a statement about the project sensors. Sensor logs enter training only after their units and calibration are written down.</p>

<h1>4. The algorithms</h1>
<p>Each description says what the method does, when this project uses it, and the scikit-learn name. Start with the default settings below. Change one setting only when you can say what you changed and why. Do not grid-search before the first honest comparison exists.</p>

<h2>4.1 Majority class, and the training mean</h2>
<p>The majority classifier ignores the features and always returns the class that appeared most often in training. The training-mean regressor always returns the average of the training targets. They are the floor. Implement them with DummyClassifier(strategy="most_frequent") and, for a number, with the mean of y_train. A crop-mean baseline is the same idea with one extra step: average the training yield of that crop, and use the global mean only when the crop was never seen in training.</p>

<h2>4.2 Persistence</h2>
<p>Used only for soil moisture. The forecast is "the next reading equals the reading we have now." At a one-day horizon, "same clock time yesterday" is the same rule, because yesterday's value at that clock is the current reading. Implement persistence by copying the latest sensor column. Do not fit it. It is the model the others have to beat. Soil water content moves slowly, so this rule is strong at 15 minutes and still strong at a few hours.</p>

<h2>4.3 Linear regression</h2>
<p>Used for yield and for moisture. It draws one straight-line combination of the features. The prediction is a weighted sum, plus a constant. It is easy to read: each coefficient says how the prediction moves when that input moves, with the others held still. It fails when the relationship bends, and when a few enormous target values dominate the fit.</p>
<p>Class: sklearn.linear_model.LinearRegression. Put StandardScaler in front of it inside a pipeline so a column measured in thousands does not drown a column measured in units. For categories such as crop and state, one-hot encode them first. One-hot encoding makes one 0/1 column per category.</p>

<h2>4.4 Logistic regression</h2>
<p>Used when the answer is a class: soil fertility, fertilizer is not required to use it, leaf colour, and crop recommendation. Despite the name, it is a classifier. It estimates the probability of each class from a weighted sum of the features, then picks the class with the highest probability. If the classes can be separated by a straight boundary in the scaled features, logistic regression does well and is still readable.</p>
<p>Class: sklearn.linear_model.LogisticRegression. Use max_iter=2000. Scale the numeric features. For a rare class, set class_weight="balanced" so the small class is not ignored. Do not read a coefficient as a field recommendation. It is a weight in this table.</p>

<h2>4.5 Decision tree</h2>
<p>Used in every classification module, and as a regressor for yield. A tree asks a sequence of yes/no questions. Example: is nitrogen above 150? If yes, is pH above 7? Each leaf of the tree holds a class, or a mean number. You can draw a shallow tree and explain it to someone else. A deep tree memorizes the training rows, including their noise, and then fails on new rows.</p>
<p>Classes: DecisionTreeClassifier and DecisionTreeRegressor. A reasonable first setting is max_depth=5 for a small table and max_depth=8 for a larger one, with min_samples_leaf=5 so a rule cannot be built on one or two rows. Set random_state=42. For classification with a rare class, set class_weight="balanced".</p>

<h2>4.6 Random forest</h2>
<p>Used in every module. A forest is many trees, each trained on a random sample of the rows and a random subset of the features. The forest's answer is the majority vote of the trees, or the average of their numbers. It is usually more stable than one tree. It is harder to draw. The useful summary is feature importance: how much each input reduced the error across the trees. Importance is not a cause. A column can look important because it stands in for another column.</p>
<p>Classes: RandomForestClassifier and RandomForestRegressor. A reasonable first setting is n_estimators=200, min_samples_leaf=5, random_state=42. Leave n_jobs at the default so a laptop uses one core. For a rare class, set class_weight="balanced" on the classifier. Trees do not need scaling. Keeping the scaler in a shared pipeline is still fine, and it makes the comparison with SVM fair.</p>

<h2>4.7 Gradient boosting</h2>
<p>Used for fertilizer and for yield. Boosting builds trees one after another. Each new tree tries to correct the errors left by the trees before it. The final prediction is the sum of the small trees. It often fits structured tables well. It also overfits a tiny table if the trees are deep or there are too many of them. scikit-learn's gradient boosting classifier has no class_weight argument, so a rare class is easier to miss than with a balanced forest.</p>
<p>Classes: GradientBoostingClassifier and GradientBoostingRegressor. A reasonable first setting on a small table is n_estimators=100, learning_rate=0.08, max_depth=2 or 3, random_state=42. learning_rate is how large a step each new tree is allowed to take. A smaller rate needs more trees.</p>

<h2>4.8 Support vector machine</h2>
<p>Used as a classifier for soil, fertilizer, leaves, and crop, and as a regressor (SVR) for moisture and yield. An SVM looks for a boundary that separates classes with the widest possible margin. With kernel="rbf" the boundary can curve. That is useful when a straight line is not enough. The cost is that the model is harder to explain, and training time grows quickly with the number of rows.</p>
<p>For a few thousand rows or fewer, use SVC(kernel="rbf", C=1.0, random_state=42). C is how hard the model tries to classify every training point. A large C memorizes. Always scale the features first. RBF on an unscaled column is meaningless. For tens of thousands of rows, use a linear SVM instead: LinearSVC for classes, LinearSVR for numbers. Say in the report that you used the linear form because the curved kernel is slow at that size. If the fit prints a convergence warning, do not treat a tiny gain over the baseline as a result. Increase max_iter once, and if it still warns, record the warning and keep the baseline.</p>

<h1>5. How to build and run a module</h1>
<h2>5.1 Folder</h2>
<p>Make one folder per module. Keep the raw file in data/raw and do not overwrite it. Write plots and tables to results/. One script, train_baselines.py, should run from start to finish.</p>
<pre>module_01_soil_nutrient/
  requirements.txt
  train_baselines.py
  data/raw/
  results/</pre>
<p>requirements.txt lists the libraries and versions you actually ran. A starting file is:</p>
<pre>pandas
numpy
scikit-learn
matplotlib
pillow</pre>
<p>Pillow is needed only for the leaf module. You may omit it from the others.</p>
<h2>5.2 Run it</h2>
<p>On macOS or Linux, from the module folder:</p>
<pre>python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python train_baselines.py</pre>
<p>On Windows, from the module folder:</p>
<pre>py -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
py train_baselines.py</pre>
<p>The script should finish without you typing anything else. Set matplotlib to a non-interactive backend at the top of the file so saving a plot does not need a window:</p>
<pre>import matplotlib
matplotlib.use("Agg")</pre>
<h2>5.3 The shape of the script</h2>
<p>Use the same shape in every tabular module. The leaf module adds one step that turns an image into numbers, then uses this same shape.</p>
<pre>SEED = 42

def load_table():
    # read the raw file, check column names, return a DataFrame
    ...

def main():
    frame = load_table()
    # print rows, missing values, duplicate rows, class counts
    x_train, x_test, y_train, y_test = train_test_split(
        features, target,
        test_size=0.20,
        random_state=SEED,
        stratify=target,   # omit this line for regression
    )
    models = {
        "majority_baseline": DummyClassifier(strategy="most_frequent"),
        "decision_tree": DecisionTreeClassifier(
            max_depth=5, min_samples_leaf=5, random_state=SEED
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200, min_samples_leaf=5, random_state=SEED
        ),
    }
    for name, model in models.items():
        pipeline = Pipeline([
            ("scale", StandardScaler()),
            ("model", model),
        ])
        pipeline.fit(x_train, y_train)
        prediction = pipeline.predict(x_test)
        # score prediction against y_test and save the row
</pre>
<p>Replace the model dictionary with the candidates for that module. For a time-ordered module, do not call train_test_split. Cut the table on the date or the year.</p>
<h2>5.4 What to save</h2>
<p>Save a comparison table as CSV, a bar chart of the main metric, a confusion matrix or a residual plot, and a short text report from sklearn.metrics.classification_report. For a forest, save a bar chart of feature importance. Also save a small JSON or text file with the seed, the row counts, the split rule, and the library versions. Do not save a large model file. The script can train again.</p>

<h1>6. Module by module</h1>

<h2>6.1 Module 1 — soil fertility class</h2>
<p><b>Question.</b> Predict the fertility class already stored in the table. Do not invent a class from a threshold of your own.</p>
<p><b>Data.</b> The 15 October assignment points at the notebook nouraalgohary/soil-fertility-prediction. Use the original table linked from that notebook, not a modified table in which extra rows were added. Columns are N, P, K, ph, ec, oc, S, zn, fe, cu, Mn, B, and fertility. The author's map is 0 = Less Fertile, 1 = Fertile, 2 = Highly Fertile. Record that you used this map.</p>
<p><b>Before you split.</b> Count exact duplicate rows. Count each class in the raw file and again after exact copies are removed. Train the comparison you will quote on the unique rows. If you also train with the copies left in, label that run invalid. A copied row in both train and test is not a test.</p>
<p><b>Split.</b> Stratified 80/20, seed 42, after the copies are removed.</p>
<p><b>Models.</b> Majority class, decision tree, logistic regression, random forest, RBF SVM. Use class_weight="balanced" on the classifiers that support it, because one class may be rare once copies are gone.</p>
<p><b>Report.</b> Accuracy, macro precision, macro recall, macro F1, and the confusion matrix. The headline number is macro F1. Add a bar chart of forest importance. Say which nutrients the forest used, and say that importance is not a cause.</p>

<h2>6.2 Module 2 — soil-moisture forecast</h2>
<p><b>Question.</b> Forecast soil water content. This dataset does not contain an irrigate / do-not-irrigate label. Do not create one.</p>
<p><b>Data.</b> Zenodo record 10.5281/zenodo.20665025, licence CC BY 4.0. It is 15-minute data from an experimental field in Gallipoli, Italy. The clean CSV has a datetime and two sensors, port 1 and port 2. The Zenodo text places odd ports at 30 cm and even ports at 60 cm. Rainfall and irrigation files are in the same zip. Leave them out of the first version. The first question is whether recent water-content readings beat persistence.</p>
<p><b>Target and features.</b> Put the series on a regular 15-minute grid. The target at time t is the reading at t plus the horizon. Features known at time t may include the current reading, a few lags (1 step, 1 hour, 4 hours, 24 hours), the mean of the last 4 hours and the last 24 hours, and the hour and day of year as sine and cosine. Drop a row if any of those values, or the future target, is missing. Do not fill a gap with a made-up number.</p>
<p><b>Horizons.</b> 15 minutes (1 step), 4 hours (16 steps), and 24 hours (96 steps). Make 4 hours the headline. Use port 1 for the full comparison. Repeat the 4-hour check on port 2.</p>
<p><b>Split.</b> Take the first 80 percent of the calendar as training and the last 20 percent as test. Use the same cutoff for every horizon. Do not shuffle.</p>
<p><b>Models.</b> Persistence. Yesterday-at-the-target-clock, which you can compute with a shift of 96 minus the horizon. Linear regression. Random forest regressor. Linear SVR, with the features scaled. Skip ARIMA in this version. Refitting it at every test time, on two years of 15-minute data, is a separate project. The linear model is already an autoregression on the lags.</p>
<p><b>Report.</b> MAE, RMSE, and R-squared at each horizon, in water-content units. A difference of 0.001 is one tenth of one percent of volumetric water content. Plot the last 14 days of the test period: recorded, persistence, and your best fitted model. Plot the residuals of that model. If the forest is not better than persistence, say so. That is a result.</p>

<h2>6.3 Module 3 — pearl-millet leaf disease</h2>
<p><b>Question.</b> Classify a leaf image. The full Zenodo record 10.5281/zenodo.17086194, licence CC BY 4.0, has five groups: healthy, Sclerospora graminicola, Exserohilum rostratum, Moesziomyces bullatus, and Pyricularia. The images were collected in Burkina Faso.</p>
<p><b>Keep the first version small.</b> The smut archive is about 230 MB and the Pyricularia archive is about 370 MB. For a normal PC, download only the three small archives: healthy, Sclerospora, and Exserohilum. Write that choice in the report. Do not describe a three-class score as a five-class result. Delete the zip after you unpack it so you do not store the images twice.</p>
<p><b>Turn an image into features.</b> Open it with Pillow, convert to RGB, resize a copy to 64 by 64, and divide pixel values by 255. Keep the mean and the standard deviation of red, green, and blue, and an 8-bin histogram of each channel. That is 30 numbers. The folder name is the label. Do not augment this version. Flips belong only on a training set, and only in a later image model.</p>
<p><b>Split.</b> Stratified 80/20, seed 42, on the 30-number table. The Zenodo folders do not name the plant. Two photos of the same leaf can fall on both sides of this split, so the score can be optimistic. Say that in the report.</p>
<p><b>Models.</b> Majority class, decision tree, logistic regression, random forest, linear SVM. Scale the features for logistic regression and the SVM.</p>
<p><b>Report.</b> The same classification metrics as Module 1, plus the class counts. The test set will be small. A perfect score on a handful of images is not a reason to stop. Read the per-class recall, especially for the class with the fewest photos. A convolutional network is optional later. It is not required to finish this module.</p>

<h2>6.4 Module 4 — fertilizer name</h2>
<p><b>Question.</b> Predict Fertilizer Name from temperature, humidity, moisture, soil type, crop type, nitrogen, potassium, and phosphorus.</p>
<p><b>Data.</b> Use the public file named in the 15 October assignment, the Fertilizer Prediction.csv linked from the Hugging Face space SamarthPujari/Crop-and-Fertilizer-Recommendation-System. Do not merge the Kaggle playground file, and do not merge any much larger file that happens to share these column names. Those are different datasets.</p>
<p><b>Before you model.</b> Strip spaces from the column names. The source spells temperature as Temparature. Rename it in the script and record the rename. Count exact duplicate rows and remove them before the split. Open the label column and look for values that look like dates. Spreadsheet software often reads an NPK grade such as 10-26-26 as a date. Map any such value back to the grade, and write the mapping in the report.</p>
<p><b>Split.</b> The cleaned table is small, and some grades have very few rows. A single 80/20 split can leave one test row in a grade. Use the same StratifiedKFold, 5 folds, shuffle=True, random_state=42, for every model. Build the confusion matrix from out-of-fold predictions, so each row is scored by a model that did not train on it. cross_val_predict does this.</p>
<p><b>Models.</b> Majority class, decision tree (max_depth=4, min_samples_leaf=3), random forest, RBF SVM, and gradient boosting with max_depth=2. Use class_weight="balanced" where the class supports it.</p>
<p><b>Report.</b> Mean and standard deviation of accuracy and macro F1 across the five folds. A high mean with a large standard deviation means one missed rare grade moves the score. Fit the forest once on all cleaned rows only to draw importance, and label that plot as descriptive. The score comes from the folds, not from that full-data fit. Expect nitrogen, phosphorus, and potassium to matter. Check whether they do.</p>

<h2>6.5 Module 5 — crop yield</h2>
<p><b>Question.</b> Predict the recorded yield of a later year. Dataset A is crop yield for Indian states and union territories, 1997 to 2020, from the Kaggle dataset akshatgupta7/crop-yield-in-indian-states-dataset. A public mirror of crop_yield.csv is acceptable if you record the mirror and the date you downloaded it. Do not merge the separate FAO country table.</p>
<p><b>Split.</b> Train on Crop_Year up to and including 2016. Test on 2017 to 2020. Do not shuffle. Count how many test rows fall in 2020. That year is thin in this file.</p>
<p><b>Leakage check, before any model.</b> Compute Production divided by Area and compare it with Yield. If they match, Production is the answer written another way. Leave Production out of the features. Also divide Fertilizer by Area and Pesticide by Area and see whether those ratios are almost constant inside a year. If they are, they are a year-level rate times area, not a soil test. Leave them out of the main model. You may run one extra forest that puts them back, on the same split, and report the difference.</p>
<p><b>Features for the main model.</b> Crop, Season, State, Crop_Year, Area, Annual_Rainfall. Strip spaces on the text columns. One-hot encode the text columns with handle_unknown="ignore", so a category that appears only in the test years does not crash the script.</p>
<p><b>Models.</b> Training-mean baseline. Crop-mean baseline. Linear regression. Decision tree regressor. Random forest regressor. Gradient boosting regressor. Linear SVR on scaled features. The two means are part of the comparison, not optional.</p>
<p><b>Report.</b> MAE, RMSE, R-squared, and median absolute error on the full test set. Then compute the same four numbers on the test rows that are not coconut, and on the Bajra rows alone. Bajra is the pearl-millet name in this table. Jowar, ragi, and small millets are related grains. Report them as their own slice if you wish, and do not call them pearl millet. Look at the largest yields in the file before you interpret R-squared. Crops in this table are not all on one physical scale. A model can look excellent because it recognises one high-valued crop. The crop-mean baseline will already capture a large part of that. The model you carry forward is the one that still helps after that fact is visible, including on Bajra. Do not delete extreme rows to make RMSE smaller. List them.</p>

<h2>6.6 Module 6 — crop recommendation</h2>
<p><b>Question.</b> Predict the crop label from N, P, K, temperature, humidity, ph, and rainfall.</p>
<p><b>Data.</b> The Kaggle dataset siddharthss/crop-recommendation-dataset. A public copy with 2,200 rows and these eight columns is the right file if the columns and the row count match. The description says the table was built by augmenting Indian rainfall, climate, and fertilizer data. It is not a set of independent field trials. Confirm the class list. This file has no pearl millet, bajra, jowar, or ragi class. Maize is not a substitute. Say that in the first paragraph of your report.</p>
<p><b>Split.</b> Stratified 80/20, seed 42. Check that each crop has the same count, and that there are no duplicate rows.</p>
<p><b>Models.</b> Majority class, decision tree, logistic regression, random forest, RBF SVM. Scale numeric features for logistic regression and the SVM.</p>
<p><b>Report.</b> Accuracy and macro F1, the confusion matrix, and forest importance. Then add one behaviour check, because a high accuracy on this file is easy to over-read. Take the training median of rice, maize, cotton, and chickpea. Hold every column at that median and change only pH from 4.5 to 8.5 in steps of 0.5. Record whether the predicted crop changes. Mark which of those pH values sit inside that crop's own training range. This check does not prove that pH never matters near a boundary. It shows what this model does on a typical profile.</p>

<h1>7. What to hand in for each module</h1>
<p>Match the 15 October deliverable.</p>
<ul>
<li>Dataset note: source, link, date you downloaded it, row count, columns, target, and the limits of the file.</li>
<li>One script that runs from start to finish, plus requirements.txt.</li>
<li>A comparison table. Every model uses the same split and the same metric definitions.</li>
<li>A confusion matrix, or a residual plot for a numeric target.</li>
<li>Three findings that point at a number or a plot you produced.</li>
<li>Two limitations. Name the dataset limit and the model limit separately.</li>
<li>Which model should be tuned next, and why. "It had the highest score" is not enough if the baseline is close, the test set is tiny, or the score is carried by one crop or one copied class.</li>
</ul>
<p>Write the findings after you look at the errors. A model that ties the simple baseline is a valid finding. A model that wins by using a leaked column is not.</p>

<h1>8. A short checklist before you trust a score</h1>
<ul>
<li>Is the target the one in the file, rather than a label you created?</li>
<li>Could any input column be used to rebuild the target?</li>
<li>Were exact duplicate rows removed before the split?</li>
<li>Was scaling fit on the training rows only?</li>
<li>For moisture and yield, is the test period later in time than the training period?</li>
<li>Does the table include a simple baseline, not only the best model?</li>
<li>For classes, did you report macro F1 and the confusion matrix, not only accuracy?</li>
<li>For numbers, did you report MAE, RMSE, and R-squared, and look at extreme values?</li>
<li>Does the written finding follow from the table, without a claim about a millet field the file does not contain?</li>
</ul>

<h1>9. Glossary</h1>
<p><b>Baseline.</b> A simple rule used as the score to beat.</p>
<p><b>Class weight.</b> A setting that makes a rare class count more during training.</p>
<p><b>Confusion matrix.</b> A table of actual class against predicted class.</p>
<p><b>Cross-validation.</b> Several train/test cuts of the same table. You report the average and the spread.</p>
<p><b>Feature importance.</b> A forest's summary of which inputs it used. Not a cause.</p>
<p><b>Horizon.</b> How far ahead a forecast looks.</p>
<p><b>Leakage.</b> Information from the answer, or from the test rows, reaching the model during training.</p>
<p><b>Macro F1.</b> The average of the per-class F1 scores. A rare class counts equally.</p>
<p><b>MAE.</b> Mean absolute error. Average size of the miss.</p>
<p><b>One-hot encoding.</b> One 0/1 column per category.</p>
<p><b>Persistence.</b> Forecast that the series will stay at its latest value.</p>
<p><b>Pipeline.</b> A scaler and a model tied together so the scaler is fit on the training rows only.</p>
<p><b>R-squared.</b> How much better you are than predicting the average. Negative means worse than that.</p>
<p><b>RMSE.</b> Root mean squared error. Large misses count more than they do in MAE.</p>
<p><b>Seed.</b> The number that makes a random split or a random forest repeatable. Use 42.</p>
<p><b>Stratified split.</b> A split that keeps each class's share in both train and test.</p>
"""


def main() -> None:
    pdf = Guide(format="A4", unit="mm")
    pdf.set_auto_page_break(auto=True, margin=16)
    pdf.set_margins(16, 22, 16)
    pdf.add_font("Arial", "", f"{FONT}/Arial.ttf")
    pdf.add_font("Arial", "B", f"{FONT}/Arial Bold.ttf")
    pdf.add_font("Arial", "I", f"{FONT}/Arial Italic.ttf")
    pdf.add_font("Arial", "BI", f"{FONT}/Arial Bold Italic.ttf")
    pdf.add_font("Mono", "", f"{FONT}/Andale Mono.ttf")
    cover(pdf)
    pdf.add_page()
    pdf.write_html(
        HTML,
        font_family="Arial",
        li_prefix_color=(22, 54, 84),
        table_line_separators=True,
        tag_styles={
            "h1": TextStyle(
                font_family="Arial",
                font_style="B",
                font_size_pt=16,
                color=(22, 54, 84),
                t_margin=4,
                b_margin=3,
            ),
            "h2": TextStyle(
                font_family="Arial",
                font_style="B",
                font_size_pt=13,
                color=(28, 78, 112),
                t_margin=3,
                b_margin=2,
            ),
            "p": TextStyle(font_family="Arial", font_size_pt=11, b_margin=2),
            "li": TextStyle(font_family="Arial", font_size_pt=11, b_margin=1),
            "pre": TextStyle(
                font_family="Mono",
                font_size_pt=8,
                fill_color=(244, 246, 248),
                color=(30, 40, 50),
                t_margin=1,
                b_margin=2,
            ),
        },
    )
    pdf.output(OUT)
    print(OUT, "pages", pdf.pages_count)


if __name__ == "__main__":
    main()
