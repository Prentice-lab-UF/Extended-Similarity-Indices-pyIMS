# -*- coding: utf-8 -*-
"""
Created on Mon May 10 15:54:37 2021

@author: nellin
"""

# import necessary modules
import csv
import numpy as np

# Define get_scores function
# Retrieves scores values and coreesponding spectra indices from all principal components in the CSV file and formats it into a 3-D numpy array
def get_scores(csv_filepath, delimiter = None):
    
    # Function description
    """
    
    Arguments
    ----------
    csv_filepath : 'str' 
        DESCRIPTION : .csv file that contains the scores values and pixel indices 
        format : (pixel id; PC1 scores; PC2 scores; . . .; PCn scores)
    
    delimiter : 'str' or None
        DESCRIPTION : define delimiter used in CSV file. Default is None which asigns the delimiter as ';'
        delimiter used must be compatible with csv.reader() function from csv import module
        
    Returns
    -------
    all_scores : numpy array of data type float64
        DESCRIPTION : numpy array of shape (n PC's, n pixels, 2)
        format of array : two columns : column-0 = pixels id's; column-1 = scores value for pixel in PC
                        each slice corresponds to a single PC in order starting from slice 0'

    """
    # Open CSV and defines as 'scores'
    with open(csv_filepath,'rt') as f:
      
      if delimiter == None:
          delimiter = ';'
          
      scores = csv.reader(f, delimiter = delimiter)
      
      # iterating over each row of the CSV file the script finds where the scores data starts
      # then begins extracting and separating the values into their corresponding slice of the 3-D numpy array
      for row in scores:
          
          # try/except functions are used to skip over file descriptions in the CSV file otherwise ValueErrors and IndexErrors arise
          try:
              # index 0 of row corresponds to the spectrum/pixel id
              # Looks for where the spectrum id is equal to 0 (indicates it is the first spectrum of the image) to start pulling out all the relevant values
              if int(row[0]) == 0:
                  # Create list of principal components to pull scores values
                  # Indices >0 in row correspond to the principal components
                  PC_index = np.arange(1, (len(row)))
                  # Create empty 3-D array to dump spectrum id and corresponding scores values
                  all_scores = np.empty((len(row)-1,0,2))
              # Create 2-D array to gather all the scores values for the single spectrum  
              PC_scores = np.empty([0,2])
              # appends all scores values and spectrum id to PC_scroes array
              for PC in PC_index: PC_scores = np.append(PC_scores, ([[float(row[0]), float(row[PC])]]), axis = 0)
              # Reshapes PC_scores to a 3-D array with 1 row and n slices where n is the number of principal components
              # Appends the reshaped PC_scores to the all_scores array so each row corresponds to a spectrum and each slice is a principal component
              all_scores = np.append(all_scores, PC_scores.reshape(max(PC_index), 1, 2), axis = 1)
          
          # If these errors arise they are skipped
          # errors arise due to the manner in which the CSV file from SCILs lab software is saved
          # first x number of lines has excess information regarding the file and the image that is not part of the scores data
          except ValueError:
              continue
          except IndexError:
              continue
    
    # Returns the 3-D numpy array containing all the spectrum ids and corresponding scores values for each principal component 
    # each slice corresponds to a principal component and each row corresponds to a spectrum
    # column 0 conatins all the spectrum ids (repeated for each slice) and column 1 contains all the scores values 
    return all_scores


# Define get_loadings() function 
# Retrieves all m/z and loadings values from all principal components in the CSV file and formats it into a 3-D numpy array 
def get_loadings(csv_filename, delimiter = None):
    # Function description
    """
    
    Arguments
    ---------- 
    csv_filepath : 'str' 
        DESCRIPTION : .csv file that contains the m/z and loadings values 
        format : (m/z; PC1 loadings; PC2 loadings; . . .; PCn loadings)
    
    delimiter : 'str' or None
        DESCRIPTION : define delimiter used in CSV file. Default is None which asigns the delimiter as ';'
        delimiter used must be compatible with csv.reader() function from csv import module

    Returns
    -------
    all_loadings : numpy array of data type float64
        DESCRIPTION : numpy array of shape (n PC's, n m/z's, 2)
        format of array : two columns : column-0 = m/z values; column-1 = loadings value for m/z in PC
                        each slice corresponds to a single PC in order starting from slice 0

    """
    # Open CSV file and defines as f
    with open(csv_filename,'rt') as f:
      # Sets delimiter to ';' if delimiter is None
      # default is None
      if delimiter == None:
          delimiter = ';'
      # sets file contents equal to 'loadings'
      loadings = csv.reader(f, delimiter = delimiter)
      # Iterates over all the rows of the CSV file looking for the line that starts with 'm/z'
      for row in loadings:
          # Try is used to ignore errors occured when iterating over the first few lines that contain file information but no loadings data
          try:
              # When 'm/z' is found the row containging the column labels for the SV file is used to create a temporary array to pull all the loadings values 
              # from every principal component for an individual m/z value
              if row[0] == 'm/z':
                  # np.arange() starts at 1 because row has n+1 number of elements where n is the number of principal components in the CSV file
                  # the extra element is the m/z value
                  PC_index = np.arange(1, (len(row)))
                  # all_loadings array is where all the loadings and m/z data will be dumped and returned
                  # a 3-D numpy array where the first column of every slice contains the m/z values and the second contains the loadings, where each slice corresponds to a principal component
                  all_loadings = np.empty((len(PC_index),0,2))
              # Looks for when 'PC_index' is defined in local variables 
              # Once it is defined, the loadings data can start being accumulated and dumped into the all_loadings array
              if 'PC_index' in locals():
                  # Makes sure the PC_loadings array is empty so it doesnt carry over data from the previous row
                  PC_loadings = np.empty([0,2])
                  # Appends the m/z and loadings values to the PC_loadings array 
                  # Each row corresponds to a principal component
                  # row 0 is PC 1, row 1 is PC 2, ... row n is PC n+1
                  for PC in PC_index: PC_loadings = np.append(PC_loadings, ([[float(row[0]), float(row[PC])]]), axis = 0)
                  # Appends the reshaped PC_loadings array to all_loadings array so that it adds row wise
                  # meaning row 0 of PC_loadings is now in slice 0 of all_loadings, row 1 of PC_loadings is now in slice 1 of all_loadings, and so on 
                  all_loadings = np.append(all_loadings, PC_loadings.reshape(max(PC_index), 1, 2), axis = 1)
          # If these errors arise they are skipped
          # errors arise due to the manner in which the CSV file from SCILs lab software is saved
          # first x number of lines has excess information regarding the file and the image that is not part of the scores data
          except ValueError:
              continue
          except IndexError:
              continue
    # Returns the 3-D numpy array containing all the m/z and corresponding loadings values for each principal component 
    # each slice corresponds to a principal component and each row corresponds to a m/z
    # column 0 conatins all the m/z (repeated for each slice) and column 1 contains all the loadings values 
    return all_loadings



# Below is used to convert the scores CSV to 3-D numpy array 
# Define the CSV file to be converted 
csv_filename = 'PCA_04-17-2019_MB_Imaging_raw_data-Scores.csv'
csv_filename_l = 'PCA_04-17-2019_MB_Imaging_raw_data-Loadings.csv'

# Define the save path where the 3-D array will be saved
save_path = 'Fingerprint_data/04-17-2019_MB_Imaging/scores/'

# Define the folder path where the CSV file is stored
folder = 'CSV_files/'

# Use function to get 3-D array by defining as some variable 
scores = get_scores(f"{folder}{csv_filename}")
loadings = get_loadings(f"{folder}{csv_filename_l}")

# Save 3-D array as .npy
output_filename = csv_filename.split(".")[0]
output_filename_l = csv_filename_l.split(".")[0]

np.save(f"{save_path}{output_filename}", scores)
np.save(f"{save_path}{output_filename_l}", loadings)

