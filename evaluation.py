import pickle  
import numpy as np
from datasets import Dataset

class Evaluation:
    
    def __init__(self, file: str, data: Dataset, threshold=0.6, white='non_aae'): 
        ''' 
        Parameters: 
            file: file of results in .p ext; 
            data: test data to be evaluated; 
            threshold: cutoff for aae identification 
        '''
        self.predictions = self.parse_result(file)  
        self.labels = self.parse_label(data) 
        self._aae_idx, self._white_idx = self.parse_demo(data, threshold, white)
        self.metrics = {} 
        assert len(self.labels) == len(self.predictions) 

    @ staticmethod 
    def parse_result(file): 
        with open(file, 'rb') as f:
            obj = pickle.load(f)
        return obj.predictions.argmax(axis=1) 
    
    @ staticmethod 
    def parse_demo(data, threshold, white): 
        props = np.array(data['demo_props']) 
        aae = np.where(props[:,0] >= threshold)[0]
        if white == 'non_aae':
             white = np.where(props[:,0] < threshold)[0]
        else: 
            white = np.where(props[:,-1] >= threshold)[0] 
        return aae, white 
                
    @ staticmethod 
    def parse_label(data):
        return np.array(data['labels']) 
    
    def eval(self): 
        self._get_EOD()
        self._get_SPD()
        self._get_DI() 
        self._get_AOD()
        return self.metrics 

    def _get_EOD(self): 
        ''' 
        calculates the 'equal opportunity difference' {
            between true positive rates of white and aae predictions; 
        } 
        ''' 
        res = {} 
        white_tpr_1 = ((self.predictions == 1)&(self.labels == 1))[self._white_idx].sum()  / (self.labels == 1)[self._white_idx].sum() 
        aae_tpr_1 = ((self.predictions == 1)&(self.labels == 1))[self._aae_idx].sum()  / (self.labels == 1)[self._aae_idx].sum()  
        white_tpr_2 = ((self.predictions == 2)&(self.labels == 2))[self._white_idx].sum()  / (self.labels == 2)[self._white_idx].sum() 
        aae_tpr_2 = ((self.predictions == 2)&(self.labels == 2))[self._aae_idx].sum()  / (self.labels == 2)[self._aae_idx].sum()  
        white_tpr = ((self.predictions != 0)&(self.labels != 0))[self._white_idx].sum()  / (self.labels != 0)[self._white_idx].sum() 
        aae_tpr = ((self.predictions != 0)&(self.labels != 0))[self._aae_idx].sum()  / (self.labels != 0)[self._aae_idx].sum()  
        res['EOD_1'] = white_tpr_1 - aae_tpr_1 
        res['EOD_2'] = white_tpr_2 - aae_tpr_2 
        res['EOD_comb'] = white_tpr - aae_tpr
        self.metrics['EOD'] = res 

    def _get_AOD(self): 
        ''' 
        calculates the 'average odds difference' {
            between the avg tpr and fpr of white and aae predictions; 
        }
        '''
        res = {} 
        eods = self.metrics['EOD'] 
        white_fpr_1 = ((self.predictions == 1)&(self.labels != 1))[self._white_idx].sum()  / (self.labels != 1)[self._white_idx].sum() 
        aae_fpr_1 = ((self.predictions == 1)&(self.labels != 1))[self._aae_idx].sum()  / (self.labels != 1)[self._aae_idx].sum()
        white_fpr_2 = ((self.predictions == 2)&(self.labels != 2))[self._white_idx].sum()  / (self.labels != 2)[self._white_idx].sum() 
        aae_fpr_2 = ((self.predictions == 2)&(self.labels != 2))[self._aae_idx].sum()  / (self.labels != 2)[self._aae_idx].sum() 
        white_fpr = ((self.predictions != 0)&(self.labels == 0))[self._white_idx].sum()  / (self.labels == 0)[self._white_idx].sum() 
        aae_fpr = ((self.predictions != 0)&(self.labels == 0))[self._aae_idx].sum()  / (self.labels == 0)[self._aae_idx].sum()
        res['AOD_1'] = 1/2 * (white_fpr_1-aae_fpr_1+eods['EOD_1']) 
        res['AOD_2'] = 1/2 * (white_fpr_2-aae_fpr_2+eods['EOD_2']) 
        res['AOD_comb'] = 1/2 * (white_fpr-aae_fpr+eods['EOD_comb']) 
        self.metrics['AOD'] = res 

    def _get_SPD(self): 
        '''
        returns the 'statistical parity difference' {
            between probabilities of toxic white and aae classifications; 
        }
        ''' 
        res = {} 
        pred_white = self.predictions[self._white_idx] 
        pred_aae = self.predictions[self._aae_idx] 
        res['SPD_1'] = ((pred_white == 1).sum()/pred_white.shape[0]) - ((pred_aae == 1).sum()/pred_aae.shape[0]) 
        res['SPD_2'] = ((pred_white == 2).sum()/pred_white.shape[0]) - ((pred_aae == 2).sum()/pred_aae.shape[0]) 
        res['SPD_comb'] = ((pred_white != 0).sum()/pred_white.shape[0]) - ((pred_aae != 0).sum()/pred_aae.shape[0]) 
        self.metrics['SPD'] = res

    def _get_DI(self): 
        '''
        returns the 'disparate impact' { 
            not impacted by annotations; 
            acceptable between 0.8 and 1.2; 
        } 
        ''' 
        res = {} 
        pred_white = self.predictions[self._white_idx] 
        pred_aae = self.predictions[self._aae_idx] 
        res['DI_non'] = ((pred_aae == 0).sum()/pred_aae.shape[0]) / ((pred_white == 0).sum()/pred_white.shape[0])
        res['DI_tox_1'] = ((pred_aae == 1).sum()/pred_aae.shape[0]) / ((pred_white == 1).sum()/pred_white.shape[0])
        res['DI_tox_2'] = ((pred_aae == 2).sum()/pred_aae.shape[0]) / ((pred_white == 2).sum()/pred_white.shape[0]) 
        res['DI_tox_comb'] = ((pred_aae != 0).sum()/pred_aae.shape[0]) / ((pred_white != 0).sum()/pred_white.shape[0]) 
        self.metrics['DI'] = res