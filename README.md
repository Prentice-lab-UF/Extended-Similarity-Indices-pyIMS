# Extended-Similarity-Indices-pyIMS
All scripts used for reproducing results shown in Extended Similarity Methods for Efficient Data Mining in Imaging Mass Spectrometry manuscript


READ ME:
The IMS image must be imported to SCiLS to generate the .sbd and .slx files. For the scripts here only the .sbd is needed and must be placed in the working directory.
It is recommended to set up specific folders within the working directory to save the resulting files from each script. Each script uses file paths to open and save files needed for the calculations and plots. 
This workflow generates 20 files for every intensity threshold tested, where each file is a different selected pixels percentage. Therefore, it is recommended to organize the files based on the name of the imaging experiment (.sbd file name) and the intensity threshold used. The .sbd file name will be used by the script to generate names for the new files. 
Avoid changing generated file names as this could cause errors in the following scripts. 
In some scripts the file name and path are defined separately to easily test multiple files withing a single folder. 
A save path will always need to be defined.

Some scripts will also use nested folders to create file names therefore see working_directory_structure.png for recommended nested folders srtucture.
 
NOTE: all scripts should be placed in the working directory parent folder only. Not within a nested folder.


The order of the scripts to run should generally be followed as so:
1.	PCA.py
2.	total_pixels_based_selection.py
3.	Binary_conversion.py
4.	Binary_MultComp_IMS_data_PCA.py
5.	plot_3D_file_loop.py
6.	Plot_2D_file_loop.py
7.	E_index_heatmap.py
8.	Binary_medoid_RR.py
