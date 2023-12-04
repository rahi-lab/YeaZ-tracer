# ``YeaZ-tracer`` -- determining the lineage of budding yeast cells

YazZ-tracer is a GUI application for precisely determining the lineage of budding yeast cells using only a segmentation mask. Having the bud neck markers, you can make lineage tracing even more precise. The method used in this lineage tracer has shown promising performance even when the image is very crowded.

## Installation
1- If you don't have conda or miniconda installed, download it from https://docs.conda.io/en/latest/miniconda.html.

2- Download the repository or clone it using ``git clone https://github.com/rahi-lab/YeaZ-tracer``

3- In the command line, navigate to the folder where you cloned YeaZ-GUI (command ``cd YeaZ-tracer``).

4- Create a conda environment using ``conda create -n yeaz_tracer python=3.9``

5- Activate the environment ``conda activate yeaz_tracer``

6- using pip, install the dependencies ``pip install -r requirements.txt``

7- install the package in developer mode ``pip install . -e``

## Run the graphical user interface

Launch the GUI using ``python -m bread.gui``.

## Lineage tracing

to do lineage tracing, you first need to add a mask file, then from the upper menu, choose 'new > guess lineage using nn'. This will show a configuration page where you can change the input parameters based on your data, or you can run it with default parameters. After finishing the lineage tracing, you can see a table created on the right side of the application. Here you can see every bud (new segmented cell), along with the time of first appearance and guessed mother. In the fourth column, you can see the confidence level of the neural network in predicting the parent of this bud. We suggest you manually check the lineage traces of a bud if the confidence is below 50 percent. If you click on a cell containing a cell number, the image zooms into the cell so you can examine it further. You can also add microscopy images to the images and see them at overlaying on the segmentation mask. 

If you have bud neck markers, you can add the file for them and use the menu 'new > guess lineage using budneck' to predict mothers using this marker. 

You can alter the lineage tracing and after finishing, you can save a csv file for each movie using 'File > save lineage'. 
