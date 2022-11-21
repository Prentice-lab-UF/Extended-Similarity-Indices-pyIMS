# -*- coding: utf-8 -*-
"""
Created on Tue May 24 17:13:04 2022

@author: nellin
"""

# Modules used by each function 
import numpy as np
import csv
import re
import h5py
import hdf5plugin 


# Define each function

# Extract raw intensity values from SCiLS Base Data (.sbd) file type
def SCiLS_raw_intensities(file_name):
    
    # Function description
    """
    
    SCiLs Base Data must be generated using the raw data when first creating the SCiLSs files
    
    Arguments 
    ---------
    file_name : 'str'
        DESCRIPTION : .sbd file that the intensities will be extracted from
    
    Returns
    -------
    raw_arr : array of data type float32
        DESCRIPTION : array of shape (n spectra, m bins)
        format : each row contains the intensity values for a single spectrum and each column contains the intensity values for a single m/z bin
        
    """
    # Import necessary modules 
    import h5py
    import hdf5plugin 
    
    # Read .sbd file 
    with h5py.File(f'{file_name}', mode="r") as h5:
        
        # Load intensity array
        raw_arr = h5["SpectraGroups/InitialMeasurement/spectra"][:]
    
    # Return raw_arr
    return raw_arr




# Extract raw mz values from SCiLs Base Data (.sbd) file type 
def SCiLS_raw_mz(file_name):
    
    # Function description
    """
    
    SCiLs Base Data must be generated using the raw data when first creating the SCiLSs files
    
    Arguments 
    ---------
    file_name : 'str'
        DESCRIPTION : .sbd file that the intensities will be extracted from
    
    Returns
    -------
    mz_range : array of data type float64
        DESCRIPTION : array of shape (1, m bins)
        format : each column contains the intensity values for a single m/z bin
        
    """
    
    # Import necessary modules
    import h5py
    import hdf5plugin
    
    # Read .sbd file 
    with h5py.File(f'{file_name}', mode="r") as h5:
    
        # Load m/z array
        mz_range = h5["SamplePositions/GlobalMassAxis/SamplePositions"][:]
    
    # Return mz_range    
    return mz_range




# Extract remapped coordinates of the SCiLs Base Data (.sbd) file type
def SCiLS_coords(file_name):
    
    # Function description
    """
    
    SCiLs Base Data must be generated using the raw data when first creating the SCiLSs files
    
    Arguments 
    ---------
    file_name : 'str'
        DESCRIPTION : .sbd file that the intensities will be extracted from
    
    Returns
    -------
    New_coords : array of data type int32
        DESCRIPTION : array of shape (n spectra, 2)
        format : column-0 contains the remapped x-coords for each pixel and column-1 contains the remapped y-coords for each pixel
        
    """

    # Import necessary modules
    import h5py
    import hdf5plugin

    # Read .sbd file 
    with h5py.File(f'{file_name}', mode = 'r') as h5:
        
        # Load raw coordinates
        coords = h5["Registrations/0/Coordinates"][:]
    
    # Remap image coordinates
    # Must be remapped otherwise image is distorted
    xcoords, ycoords = coords[0], coords[1]
    new_xcoords, new_ycoords = xcoords.copy(), ycoords.copy()
    
    for i, x in enumerate(np.unique(xcoords)):
        new_xcoords[new_xcoords == x] = i
    for i, y in enumerate(np.unique(ycoords)):
        new_ycoords[new_ycoords == y] = -i-1
    
    new_xcoords = new_xcoords.astype(np.int32)
    new_ycoords = new_ycoords.astype(np.int32)
    new_coords = np.array([new_xcoords, new_ycoords]).transpose()
    
    # Return new_coords
    return new_coords




# Extract mz values from .mir and .csv file types
def get_mz(filename, convert = None, include = "all"):
    
    # Function description
    """
    
    Arguments
    ----------
    filename : 'str'
        DESCRIPTION : .mir or .csv file types (.csv must be from SCiLs software export) to extract m/z form
    
    convert : 'dictionary' or 'numpy'
        DESCRIPTION : Must choose what to convert .mir file type to either a dictionary or numpy array. CSV can only be converted to numpy array.
        format : Single peak values are not extracted, instead they are extracted as upper and lower mz bounds for each peak. 
            For numpy format with only mz selected. column-0 contains the lower bounds for each peak and column-1 contains the upper bounds for each peak.
        
    include : 'all' or 'mz only'
        DESCRIPTION : 'all' is set by default and only applies to .mir file types to extract the rest of the parameters for each peak. The remainder of the information includes: 
            Type, Name, Color, Show, MinIntensity, IntensityThreshold, AbsIntens, LogScale, MinMass, MaxMass, Integrate, FindMass, and RelMass
            'mz only' will only extract the upper and lower limits of the selected peaks
                
    Returns
    -------
    mz_list : dictionary or numpy array
        DESCRIPTION : dictionary or numpy array contianing the m/z information stored in the .mir or .csv file formats.
    
    """
    
    # Import necessary modules
    import numpy as np
    import re 
    import csv
    
    # Raises TypeError if convertion type not chosen 
    if convert == None:
        raise TypeError("Must choose to convert file to dictionary or numpy array. If file type is CSV, only numpy can be chosen.")
    
    # Determine what the file type is
    file_type = filename.split(".")[1]
    
    # For CSV files
    if file_type == "csv" or file_type == "CSV":
        
        # Raises TypeError if CSV is to be converted to dictionary
        if convert == 'dictionary':
            raise TypeError("CSV file type cannot be converted to dictionary")
        
        # Reads the CSV file if numpy is chosen 
        if convert == 'numpy':
            with open(filename,'rt') as f:
              mzs = csv.reader(f, delimiter = ';')
              
              # Iterate across the CSV file by row 
              for row in mzs:
                  
                  # Tries to append the mz values to list 
                  # ValueErrors and IndexErrors are ignored to parse out the first few lines that contain general file information
                  try:
                      # Row is pulled as single str() argument and must be split by ',' to call correctly
                      row = row[0].split(',')

                      # Creates the mz lists once the column labels are found indicating that the next lines will contain the actual mz values
                      if row[0] == 'm/z':
                          mz_lower = []
                          mz_upper = []
                      if 'mz' in locals() and row[0] != 'm/z':
                          mz_lower.append(float(row[0]-row[1]))
                          mz_upper.append(float(row[1]+row[1]))                  
                  except ValueError:
                      continue
                  except IndexError:
                      continue
            
            # Create mz_list to be returned
            mz_list = np.array([mz_lower, mz_upper]).transpose()
    
    # For .mir files 
    if file_type == "mir":
        
        # Set split parameters for each row of the file
        patern = "Type=|Name=|Color=|Show=|MinIntensity=|IntensityThreshold=|AbsIntens=|LogScale=|MinMass=|MaxMass=|Integrate=|FindMass=|RelMass="
        
        # Open and read .mir file
        with open(filename, "r") as f:
            
            # For convertion to dictionary
            if convert == 'dictionary':
                
                # For including all the data in the .mir file 
                if include == 'all':
                    
                    # Create dictionary
                    mz_list = {}
                    for row in f:
                        
                        # Split row according to patern
                        row = row.replace('"', '')
                        row = re.split(patern, row)
                        
                        # Selects row that has the data
                        if row[0] == "<Result ":
                            
                            # Uses name of the peak from .mir for dictionary keys and then appends all the info into the key with a nested dicitonary 
                            mz_list[str(row[2])] = {'Color' : row[3],
                                                    'Show' : row[4],
                                                    'MinIntensity' : row[5],
                                                    'IntensityThreshold' : row[6],
                                                    'AbsIntensity' : row[7],
                                                    'LogScale' : row[8],
                                                    'MinMass' : row[9], 
                                                    'MaxMass': row[10],
                                                    'Integrate' : row[11],
                                                    'FindMass' : row[12]}
                
                # For only extract mz values to from .mir to convert to dictionary
                # Follows same method as above but only adds the upper and lower bounds of the selected peak
                if include == 'mz only':
                    mz_list = {}
                    for row in f:
                        row = row.replace('"', '')
                        row = re.split(patern, row)
                        if row[0] == "<Result ":
                            mz_list[str(row[2])] = {'MinMass' : row[9], 
                                                    'MaxMass': row[10]}
                            
            # For conversion to numpy array 
            if convert == 'numpy':
                
                # No lables are included 
                # Output is a 2D array 
                # Each row contains the data for each peak in the .mir file
                if include == 'all':
                    mz_list = np.empty([0, 11])
                    for row in f:
                        row = row.replace('"', '')
                        row = re.split(patern, row)
                        if row[0] == "<Result ":
                            mz_list = np.append(mz_list, row[2:13])
                    mz_list = np.reshape(mz_list, (len(mz_list)//11, 11))
                
                # For converting only the mz values to numpy array 
                # Follows same method as above but only contains the 
                if include == 'mz only':
                    mz_list = np.empty([0,2])
                    for row in f:
                        row = row.replace('"', '')
                        row = re.split(patern, row)
                        if row[0] == "<Result ":
                            mz_list = np.append(mz_list, [row[9],row[10]])
                    mz_list = np.reshape(mz_list, (len(mz_list)//2, 2))

    # Returns mz_list as output
    return mz_list




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
    
    # Import necessary modules     
    import csv
    import numpy as np

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
    
    # Import necessary modules    
    import csv
    import numpy as np

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




# Define normalization function to normalize MS datasets from SCiLs_raw_intensities() function
def normalize(raw_spc, function = "RMS"):
    
    # Function description
    """
    
    Arguments
    ----------
    raw_spc : array from SCiLs_raw_intensities() function output
        DESCRIPTION : 2D array where each row is a spectrum and each column is an m/z bin (n spectra, m bins)
            Recommended to use the raw output from the SCiLs_raw_intensities() function
            
    function : 'str'
        DESCRIPTION : define the normalization method to be used. Options are only 'RMS'(Root Mean Square) or 'TIC' (Total Ion Count)
    
    Returns
    -------
    norm_spc : 2D array 
        DESCRIPTION : 2D data matrix containing the normalized intensities.
        format : Data matrix where each row makes up the intensities for a single spectrum and each column  is a single m/z bin. Shape of the matrix is in the following format (n spectra, m bins).
       
    """
    
    #Import necessary modules
    import numpy as np
    
    # Creat norm_spc array and set number of bins
    norm_spc = raw_spc
    l = len(raw_spc.transpose())
    
    # Calculate normalized intensities based on the two methods available
    if function == "RMS":
        for i, pixel in enumerate(norm_spc):
            sum_squared = np.sum(pixel**2)
            mean_square = sum_squared/l
            RMS = mean_square**(1/2)
            pixel/=RMS
    if function == "TIC":
        for i, pixel in enumerate(norm_spc):
            TIC = np.sum(pixel)
            pixel/=TIC 
            
    # Return the normalized spectra
    return norm_spc




# Extract spectra indices for CSV exported regions from SCiLs lab software 
def get_region_indices(csv_filename):
    
    # Function description
    
    """
    
    Arguments
    ----------
    csv_filename : 'str'
        DESCRIPTION : File path and name of the CSV file for the exported region from SCiLs lab software.
    
    Returns
    -------
    region_indices : list
        DESCRIPTION : list of the spectrum indices that make up the image region exported from SCiLs.
    
    """
    
    # Import necessary modules
    import numpy as np
    import csv

    # create list
    region_indices = []
    
    # import CSV file and pull Spectrum IDs/pixel IDs
    # Ignores ValueErrors to parse out the extra irrelevant data in the file
    with open(csv_filename,'rt') as f:
      spots = csv.reader(f, delimiter = ';')
      for row in spots:
          try:
              region_indices.append(int(row[0]))
          except ValueError:
              continue
  
    # Returns region_indices list 
    return region_indices
