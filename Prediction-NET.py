"""
National Data Science Bowl:
Predict ocean health, one plankton at a time
author: Elena Cuoco
"""
import pandas as pd
import numpy as np
import datetime
import sys
from datetime import datetime
import csv
#from memory_profiler import profile
#joblib library for serialization
from sklearn.externals import joblib
import os,shutil
#json library for settings file
import json
##Read configuration parameters
file_dir = './settings-nb.json'
config = json.loads(open(file_dir).read())
MODEL_PATH=config["HOME"]+config["MODEL_PATH"]
test_file=config["HOME"]+config["TEST_DATA_PATH"]+'test.csv'
SUBMISSION_PATH=config["HOME"]+config["SUBMISSION_PATH"]
seed= int(config["SEED"])
path_to_delete =SUBMISSION_PATH
##clean old submission
if os.path.exists(path_to_delete):
    shutil.rmtree(path_to_delete)
os.mkdir(SUBMISSION_PATH)

###############################################################################
# Main
###############################################################################
####---ordered labels encoded-----------------###################

labels=['acantharia_protist', 'acantharia_protist_big_center', 'acantharia_protist_halo', 'amphipods', 'appendicularian_fritillaridae', 'appendicularian_s_shape', 'appendicularian_slight_curve', 'appendicularian_straight', 'artifacts', 'artifacts_edge', 'chaetognath_non_sagitta', 'chaetognath_other', 'chaetognath_sagitta', 'chordate_type1', 'copepod_calanoid', 'copepod_calanoid_eggs', 'copepod_calanoid_eucalanus', 'copepod_calanoid_flatheads', 'copepod_calanoid_frillyAntennae', 'copepod_calanoid_large', 'copepod_calanoid_large_side_antennatucked', 'copepod_calanoid_octomoms', 'copepod_calanoid_small_longantennae', 'copepod_cyclopoid_copilia', 'copepod_cyclopoid_oithona', 'copepod_cyclopoid_oithona_eggs', 'copepod_other', 'crustacean_other', 'ctenophore_cestid', 'ctenophore_cydippid_no_tentacles', 'ctenophore_cydippid_tentacles', 'ctenophore_lobate', 'decapods', 'detritus_blob', 'detritus_filamentous', 'detritus_other', 'diatom_chain_string', 'diatom_chain_tube', 'echinoderm_larva_pluteus_brittlestar', 'echinoderm_larva_pluteus_early', 'echinoderm_larva_pluteus_typeC', 'echinoderm_larva_pluteus_urchin', 'echinoderm_larva_seastar_bipinnaria', 'echinoderm_larva_seastar_brachiolaria', 'echinoderm_seacucumber_auricularia_larva', 'echinopluteus', 'ephyra', 'euphausiids', 'euphausiids_young', 'fecal_pellet', 'fish_larvae_deep_body', 'fish_larvae_leptocephali', 'fish_larvae_medium_body', 'fish_larvae_myctophids', 'fish_larvae_thin_body', 'fish_larvae_very_thin_body', 'heteropod', 'hydromedusae_aglaura', 'hydromedusae_bell_and_tentacles', 'hydromedusae_h15', 'hydromedusae_haliscera', 'hydromedusae_haliscera_small_sideview', 'hydromedusae_liriope', 'hydromedusae_narco_dark', 'hydromedusae_narco_young', 'hydromedusae_narcomedusae', 'hydromedusae_other', 'hydromedusae_partial_dark', 'hydromedusae_shapeA', 'hydromedusae_shapeA_sideview_small', 'hydromedusae_shapeB', 'hydromedusae_sideview_big', 'hydromedusae_solmaris', 'hydromedusae_solmundella', 'hydromedusae_typeD', 'hydromedusae_typeD_bell_and_tentacles', 'hydromedusae_typeE', 'hydromedusae_typeF', 'invertebrate_larvae_other_A', 'invertebrate_larvae_other_B', 'jellies_tentacles', 'polychaete', 'protist_dark_center', 'protist_fuzzy_olive', 'protist_noctiluca', 'protist_other', 'protist_star', 'pteropod_butterfly', 'pteropod_theco_dev_seq', 'pteropod_triangle', 'radiolarian_chain', 'radiolarian_colony', 'shrimp-like_other', 'shrimp_caridean', 'shrimp_sergestidae', 'shrimp_zoea', 'siphonophore_calycophoran_abylidae', 'siphonophore_calycophoran_rocketship_adult', 'siphonophore_calycophoran_rocketship_young', 'siphonophore_calycophoran_sphaeronectes', 'siphonophore_calycophoran_sphaeronectes_stem', 'siphonophore_calycophoran_sphaeronectes_young', 'siphonophore_other_parts', 'siphonophore_partial', 'siphonophore_physonect', 'siphonophore_physonect_young', 'stomatopod', 'tornaria_acorn_worm_larvae', 'trichodesmium_bowtie', 'trichodesmium_multiple', 'trichodesmium_puff', 'trichodesmium_tuft', 'trochophore_larvae', 'tunicate_doliolid', 'tunicate_doliolid_nurse', 'tunicate_partial', 'tunicate_salp', 'tunicate_salp_chains', 'unknown_blobs_and_smudges', 'unknown_sticks', 'unknown_unclassified']
header=labels

class AdjustVariable(object):
    def __init__(self, name, start=0.03, stop=0.001):
        self.name = name
        self.start, self.stop = start, stop
        self.ls = None

    def __call__(self, nn, train_history):
        if self.ls is None:
            self.ls = np.linspace(self.start, self.stop, nn.max_epochs)

        epoch = train_history[-1]['epoch']
        new_value = np.float32(self.ls[epoch - 1])
        getattr(nn, self.name).set_value(new_value)
class EarlyStopping(object):
    def __init__(self, patience=100):
        self.patience = patience
        self.best_valid = np.inf
        self.best_valid_epoch = 0
        self.best_weights = None

    def __call__(self, nn, train_history):
        current_valid = train_history[-1]['valid_loss']
        current_epoch = train_history[-1]['epoch']
        if current_valid < self.best_valid:
            self.best_valid = current_valid
            self.best_valid_epoch = current_epoch
            self.best_weights = [w.get_value() for w in nn.get_all_params()]
        elif self.best_valid_epoch + self.patience < current_epoch:
            print("Early stopping.")
            print("Best valid loss was {:.6f} at epoch {}.".format(
                self.best_valid, self.best_valid_epoch))
            nn.load_weights_from(self.best_weights)
            raise StopIteration()

#serialize training
#------------open model file------------------################
model_file=MODEL_PATH+'model-clf.pkl'
clf = joblib.load(model_file)
submission=SUBMISSION_PATH+'prediction-bowl.csv'
maxPixel=64 ###to be included in json settings
# prepare data
#@profile
def clean_data(data):
    clean=data.drop(['image'], axis=1)#remove image and label columns
    X=np.asarray(clean.astype(np.float32))
    ######## preprocessing
    X = X.reshape(-1, 1, maxPixel,maxPixel)
    return  X
reader_test = pd.read_table(test_file, sep=',', chunksize=1000,header=0)

with open(submission, 'a') as outfile:
    i=0
    for test in reader_test:
      X_test=clean_data(test)
      y_pred = clf.predict_proba(X_test)
      images=test['image']
      df = pd.DataFrame(y_pred, columns=labels, index=images)
      df.index.name = 'image'
      df = df[header]
      if i==0:
       df.to_csv(outfile,float_format='%.10e')
      else:
        df.to_csv(outfile,header=None,float_format='%.10e')
      i+=1
      if i%10==0:
        print i

