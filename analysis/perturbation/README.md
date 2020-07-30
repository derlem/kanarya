# The Effect of Various Perturbation Strategies on Determining the Usage of Turkish Clitics

The effect of various perturbation strategies on determining the usage of Turkish "de/da" clitics will be discussed throughout this document. 

Currently we have several models that has around %86 F1 Score and nearly %78 accuracy. We are interested in the effect of various perturbations in a sentence to the predictions of our model.

**Model:** We train a sequence tagging model over a large dataset. Each word in a sentence in this dataset is either labeled as **'O'** or **'B-ERR'** which indicates there is erroneous and not erroneous usage respectively. For instance:

Bugün O<br/>
ev O<br/>
de B-ERR<br/>
değilim O<br/>
. O<br/>

## Basic algorithm to obtain knowledge out of perturbations:

* For each sentence in the given dataset:<br/>
    * Find a word which includes a "de/da" clitic in it. This word is called **word<sub>deda</sub>**.<br/>
    * For the **word<sub>deda</sub>**, calculate in which probability the model predicts the label of it. This probability is **P<sub>0</sub>**<br/>
    * For each **perturbation<sub>i</sub>**:<br/>
        * Apply the **perturbation<sub>i</sub>** and obtain a new sentence.<br/>
        * Calculate in which probability the model predicts the label of the **word<sub>deda</sub>** in the new sentence. This probability is **P<sub>i</sub>**.<br/>
        * Calculate **Delta_P<sub>i</sub> =  P<sub>i</sub> - P<sub>0</sub>** and store it.<br/>
* Calculate the average of **Delta_P<sub>i</sub>'s** over all sentences.<br/>

The average **Delta_P<sub>i</sub>'s** for each perturbation strategy indicates the effect of each perturbation strategy  to the prediction of the model. If the **Delta_P** for a particular strategy is high, then this means that that perturbation is important for determining the "de/da" usage in a sentence.


As a starting point, I have implemented 3 simple/dummy perturbation strategies and examined their **Delta_P's**.

### Perturbation Strategy 1:

- Delete the last character of the preceding word.

**Example:**

* **Original sentence:** Bugün okul da değilim.<br/>
* **After perturbation:** Bugün oku da değilim.

**Delta P:** -0.0016(over ~1000 sentences)

### Perturbation Strategy 2:

- Delete the first character of the preceding word.

**Example:**

* **Original sentence:** Bugün okul da değilim.<br/>
* **After perturbation:** Bugün kul da değilim.

**Delta P:** 0.0014 (over ~1000 sentences)

### Perturbation Strategy 3:

- Capitalize the first character of the preceding word.

**Example:**

* **Original sentence:** Bugün okul da değilim.<br/>
* **After perturbation:** Bugün Okul da değilim.

**Delta P:** -0.00 (over ~1000 sentences)


### Further Work and Need for New Ideas

At this point, we plan to come up with new perturbation strategies and examine their **Delta P's** and try to find strategies whose **Delta P** is relatively high or low compared to the average. All ideas are welcome regarding this document. Particularly, we are looking for interesting perturbation strategies.

##### Code: https://github.com/derlem/kanarya/blob/perturbation/perturbation/perturb.py
