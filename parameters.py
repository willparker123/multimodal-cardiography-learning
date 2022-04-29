import os

"""# Global Variables / Paths"""

"""## Edit These"""
input_physionet_folderpath = "physionet-data/training-a"
input_ecgpcgnet_folderpath = "ecg-pcg-data"
input_ephnogram_folderpath = "ephnogram-data"
input_ephnogram_data_foldername = "WFDB"
input_ephnogram_target_filename = "ECGPCGSpreadsheet.csv"
output_folderpath = "data"
drive_folderpath = "Colab Notebooks"
useDrive = False

sample_rate_ecg = 2048
sample_rate_pcg = 2048
clip_len = 5 # Length of clip in seconds


"""## DO NOT EDIT These"""
drivepath = 'drive/MyDrive/'+drive_folderpath+"/"
inputpath_physionet = drivepath+input_physionet_folderpath+"/" if useDrive else input_physionet_folderpath+"/"
inputpath_ecgpcgnet = drivepath+input_ecgpcgnet_folderpath+"/" if useDrive else input_ecgpcgnet_folderpath+"/"
inputpath_ephnogram_data = drivepath+input_ephnogram_folderpath+"/" if useDrive else input_ephnogram_folderpath+"/"+input_ephnogram_data_foldername+"/"
inputpath_ephnogram_target = drivepath+input_ephnogram_folderpath+"/" if useDrive else input_ephnogram_folderpath+"/"+input_ephnogram_target_filename
outputpath = drivepath+output_folderpath+"/" if useDrive else output_folderpath+"/"
#drivepath = 'drive\\MyDrive\\'+drive_folderpath+"\\"
#inputpath = drivepath+input_folderpath+"\\" if useDrive else input_folderpath+"\\"
#outputpath = drivepath+output_folderpath+"\\" if useDrive else output_folderpath+"\\"

"""# Inside visualisation_functions, use these"""

def get_filtered_df(df, column, value):
    df = df[df[column] == value]
    # Returns a df where all values of a certain column are a certain value
    return df

def create_new_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)
        return True
    else:
        # Only create the folder if it is not already there
        return False

#def get_play_description_from_number(csv, play_no, game_id):
#    play = get_filtered_df(csv, 'playId', play_no)
#    specific_play = get_filtered_df(play, 'gameId', game_id)
#    description = specific_play['playDescription']
#    # Extracting the play description if we are to save the animation locally
#    return description
