# Designing selective kinase inhibitors

## Problem Statement:

The goal is to develop a model that can predict the pKi of a given compound against JAK1, JAK2, JAK3 and TYK2 kinases, which can then be used in identifying selective inhibitors. Selectivity in this scenario means a compound has a high pKi value for only one of the four kinases. For example, a selective compound would bind to JAK1 with a high pKi value, but with low pKi values to the other three kinases. See [problem context](#problem-context) for more detailed information.


### Sections
- **Exploratory Data Analysis**: Analyze the data `kinase_JAK.csv` and present to us your understanding of the data. Discuss any noteworthy aspects of the data and any assumptions you make.
- **Data Processing**: Build a data processing pipeline and set up your model evaluation strategies based on this specific type of data. Carefully consider how you split the data and how that may impact your model evaluation.
- **Model Development**: Implement an interesting deep learning model (e.g. transformer or graph neural net) that predicts pKi as a function of molecular structure for all four kinases. For text-based models, a tokenizer is provided in the notebook `kinase_challenge_data.ipynb`.
- **Model Evaluation**: Evaluate your model using performance metrics and relating it back to the [problem](#problem-statement)
- **Model Deployment**: Discuss how your model could be deployed as a service.
- **Future Ideas**: Consider which directions would be most interesting to pursue if you had more time and/or resources.

### Other Ideas
Based on your skills and interest, you could extend beyond the [must haves](#must-haves) in these potential areas:
- **Advanced Modeling**: Test additional ideas to create an even better model.
- **Model Comparison**: Test multiple ideas head-to-head.
- **Example Deployment**: Build out a prototype deployment solution.
- **Extend any of the Must-Haves**: Go into more depth on any of the above sections.

### Additional Notes
- Provide a detailed summary of your approach and results in either markdown or a jupyter notebook. Document your efforts even when they are not successful.
- Do your best to keep your code and outputs organized.
- Chemistry knowledge is not required. If you are unfamiliar, [`rdkit`](https://www.rdkit.org/docs/index.html) is a decent place to start.

## Problem context

Many diseases are caused by proteins in the body being too inactive or too active. As such, most drugs work by stopping or activating proteins. Here we will focus on stopping proteins (so-called inhibitors).

The way proteins work is akin to a key opening a lock - a protein can be thought of as a large object with a small keyhole. The goal of a medicinal chemist is to find a molecule - a key - that can slot perfectly into the keyhole.

However, biology doesn't like medicinal chemists. Many proteins share similar "keyholes". A key that switches off the desired protein could also switch off other similar proteins. Whilst shutting down the disease-causing protein is good, shutting down other essential proteins is bad. That's why drugs have side effects.

In the drug discovery process, medicinal chemists will typically want to design molecules that are "selective" - i.e. it only interacts with the targeted protein and nothing else. This is challenging because it is hard to know, by eye, what are the molecular features that cause binding to one protein, but not other very similar proteins.

That's where machine learning becomes useful!

If we can build models that predict the ability of a molecule to stop different proteins, we can design molecules that can selectively stop the target protein but not others.

In this problem, we will focus on the JAK (Janus-associated kinase) family of proteins. This family has 4 similar proteins: JAK1, JAK2, JAK3 and TYK2. The JAK family is strongly associated with diseases such as cancer and inflammation.

There are already several approved JAK inhibitors to treat human diseases, including tofacitinib (an unselective JAK inhibitor from Pfizer) for rheumatoid arthritis, ruxolitinib (a selective JAK1/2 inhibitor from Incyte) for intermediate- or high-risk myeloproliferative neoplasms, baricitinib (a selective JAK1/2 inhibitor) for the treatment of rheumatoid arthritis, and peficitinib (an unselective JAK inhibitor) approved in Japan for treatment of rheumatoid arthritis. However, there is a strong hypothesis that the side-effects of these drugs are caused by hitting JAK2 as collateral damage. Designing selective JAK inhibitors has been a major focus in the pharma industry.

For example, here’s a recent table from a paper from Pfizer summarizing medicinal chemists’ progress in finding selective inhibitors against JAK1 (full text of the paper available [here](https://www.osti.gov/biblio/1526050); don’t worry about reading it unless you are interested). IC50 is the concentration of compound required to shut down 50% of protein function. As such, a lower IC50 implies a more potent compound. This table shows that the chemists have found a compound that is 63x more potent against JAK1 than JAK2. That’s great, except it was a long process of trial-and-error.

![Pfizer table](pfizer_table.png)

In this problem, you are given a dataset of measured activity against 4 related proteins: JAK1, JAK2, JAK3 and TYK2. There are two related types of measurements: pIC50 and pKi – in both cases, a higher number suggests stronger binding.

## Data

The Jupyter notebook will show you how to manipulate the data `kinase_JAK.csv`. Using `rdkit` is a good way to get started working with molecules.
