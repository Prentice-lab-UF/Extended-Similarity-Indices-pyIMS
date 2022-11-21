# -*- coding: utf-8 -*-
"""
Created on Sun Jan 30 17:29:50 2022

@author: nellin
"""

# imports
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import matplotlib.image as mpimg


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
                            mz_list = np.append(mz_list, np.array([row[9],row[10]], dtype = 'float64'))
                    mz_list = np.reshape(mz_list, (len(mz_list)//2, 2))

    # Returns mz_list as output
    return mz_list


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
            

# Define name of .sbd file to extract raw intensity values from
name = "2022-02-22_NE_MB_posMS_Lipids_DAN_spray.sbd"

# Define path for folder containing the .sbd file
folder = 'SBD_files/'

# Define .mir file path/name to retreive peaks chosen for PCA calculation
mz_filename = 'PCA mz data/2022-02-22_MB_Lipids_PCA_Peaks.mir'

# Define save path for PCA results
save_path = 'Fingerprint_data/2022-02-22_NE_MB_posMS_Lipids_DAN_spray/scores/'

# Pre-processing normalization for PCA calculations
# If none required leave defined as None
norm = "RMS"

#Set number of components to calculate
components = 5



# Autogenerate output file name
output_filename = f'{name.split(".")[0]}_{str(norm)}_PCA-{str(components)}'

# Retreive mz ranges for calculation
mzs = get_mz(mz_filename, convert = 'numpy', include = 'mz only')

# Load mz values from image
all_mz = SCiLS_raw_mz(folder+name)

# Load raw intensities
raw_int = SCiLS_raw_intensities(folder+name)

# Pre-processing normalization
if not norm == None:
    raw_int = normalize(raw_int, function = "RMS")

# Load coordinates for image
coords = SCiLS_coords(folder+name)

# Create temp mz list 
mz_list = []
mz_list_i = []

# Create list of mz indices for intensity selection in raw_int for PCA calculation
for mz in mzs:
    
    select_mz_i = np.array(np.where((all_mz >= mz[0]) & (all_mz <= mz[1])))
    select_mz = all_mz[select_mz_i]
        
    mz_list_i.append(select_mz_i)
    mz_list.append(select_mz)

# create PCA input data array
pca_input = np.zeros((len(raw_int), len(mzs)))

# replace zeros with averaged intesntiy values across error ranges from mz list
for x, bounds in enumerate(mz_list_i):
    
    # averages the intenstiy values and then replaces the zeros with the averaged values
    mz_int = raw_int[:, bounds.flatten()]
    pca_input[:, x] = np.max(mz_int, axis = 1)
    
# Calculate PCA and extract results
pca = PCA(svd_solver = 'randomized', n_components = components, iterated_power = 10000)
pca_output = pca.fit_transform(pca_input)
loadings = pca.components_.reshape(components,len(mzs),1)
loadings = np.insert(loadings, 0, np.average(mzs, axis = 1), axis = 2)
explained_variance = pca.explained_variance_ratio_


# Plot and save spatial-expresion images of scores valaues
for pcx, PC in enumerate(pca_output.transpose()):
    
    img = np.zeros([np.absolute(coords[:,1]).max()+1, coords[:,0].max()+1])
    
    plt.figure(pcx, dpi=500)
    
    img_bckrnd = min(pca_output[:, pcx])
    
    img[:,:] = img_bckrnd
    
    for i,c in enumerate(coords):
        img[c[1],c[0]] = pca_output[i, pcx]
            
    title = output_filename.split("_")
    title = "_".join(title[0:-1])+f'_spatial-expression_image_PC{pcx+1}'
         
    plt.title(f"{title}")
    plt.xticks([])
    plt.yticks([])
    mpimg.imsave(f'{save_path}{title}.jpg', img.repeat(25, axis=1).repeat(25, axis=0))
    plt.imshow(img)
   
# Reshape PCA output to shape needed for total_pixels_based_selection.py    
Scores = np.reshape(pca_output, (1, len(pca_output), components))
index = np.arange(len(pca_output)).repeat(components).reshape(1, len(pca_output),components)
Scores = np.append(index, Scores, axis = 0).transpose()

# Check if reshape is done correctly 
# ValueError is raised if the reshape is not done correctly 
for pcx, PC in enumerate(pca_output.transpose()):
    if np.array_equal(Scores[pcx,:,1], pca_output[:,pcx]) == False:
        raise ValueError

# Saves PCA results
np.save(f"{save_path}{output_filename}-Scores", Scores)
np.save(f"{save_path}{output_filename}-Loadings", loadings)
np.save(f"{save_path}{output_filename}-Explained_Variance", explained_variance)