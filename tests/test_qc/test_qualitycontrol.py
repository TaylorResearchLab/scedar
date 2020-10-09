import numpy as np
#import matplotlib as mpl
#import matplotlib.pyplot as plt
#import seaborn as sns
import scipy.sparse as spsp
import pytest
#import pandas as pd

#import scedar.cluster as cluster
from scedar.qc import QualityControl


def mat_to_csc(mat, verbose=0):
        if isinstance(mat,pd.core.frame.DataFrame):
            if verbose: print('DataFrame converted to CSC matrix.')
            return spsp.csc_matrix(mat.values)
        elif  isinstance(mat,spsp.csr.csr_matrix):
            if verbose: print('CSR matrix converted to CSC matrix.')
            return spsp.csc_matrix(mat)
        elif isinstance(mat,spsp.coo.coo_matrix):
            if verbose: print('COO matrix converted to CSC matrix.')
            return spsp.csc_matrix(mat) #mat.tocsc()
        

def get_test_genes():    
    return  ['ENSG00000243485','ENSG00000237613','ENSG00000186092','ENSG00000238009','ENSG00000239945','ENSG00000237683',
                  'ENSG00000239906','ENSG00000241599','ENSG00000228463','ENSG00000237094','ENSG00000235249','ENSG00000236601',
                  'ENSG00000236743','ENSG00000231709','ENSG00000239664','ENSG00000230021', 'ENSG00000223659',
                  'ENSG00000185097', 'ENSG00000235373','ENSG00000240618','ENSG00000229905','ENSG00000010292', 'ENSG00000011426', #ENSG00000228327','ENSG00000237491', 
                  'ENSG00000129055','ENSG00000177757','ENSG00000225880','ENSG00000230368','ENSG00000269308','ENSG00000272438',
               'ENSG00000230699','ENSG00000210049','ENSG00000211459','ENSG00000097007','ENSG00000210082', 'ENSG00000241180',
                  'ENSG00000223764','ENSG00000187634','ENSG00000268179','ENSG00000188976',
               'ENSG00000187961']

oldtestgenes = ['ENSG00000243485','ENSG00000237613','ENSG00000186092','ENSG00000238009','ENSG00000239945','ENSG00000237683',
                  'ENSG00000239906','ENSG00000241599','ENSG00000228463','ENSG00000237094','ENSG00000235249','ENSG00000236601',
                  'ENSG00000236743','ENSG00000231709','ENSG00000239664','ENSG00000230021', 'ENSG00000223659',
                  'ENSG00000185097', 'ENSG00000235373','ENSG00000240618','ENSG00000229905','ENSG00000010292', 'ENSG00000011426', #ENSG00000228327','ENSG00000237491', 
                  'ENSG00000269831','ENSG00000177757','ENSG00000225880','ENSG00000230368','ENSG00000269308','ENSG00000272438',
               'ENSG00000230699','ENSG00000210049','ENSG00000211459','ENSG00000210077','ENSG00000210082', 'ENSG00000241180',
                  'ENSG00000223764','ENSG00000187634','ENSG00000268179','ENSG00000188976',
               'ENSG00000187961']  
 
def get_test_barcodes():    
    return ['AAACATACAACCAC-1', 'AAACATTGAGCTAC-1', 'AAACATTGATCAGC-1','AAACCGTGCTTCCG-1', 'AAACCGTGTATGCG-1', 'AAACGCACTGGTAC-1',
                       'AAACGCTGACCAGT-1', 'AAACGCTGGTTCTT-1', 'AAACGCTGTAGCCA-1','AAACGCTGTTTCTG-1', 'AAACTTGAAAAACG-1', 'AAACTTGATCCAGA-1',
                       'AAAGAGACGAGATA-1', 'AAAGAGACGCGAGA-1', 'AAAGAGACGGACTT-1','AAAGAGACGGCATT-1', 'AAAGATCTGGGCAA-1', 'AAAGCAGAAGCCAT-1',
                        'AAAGCAGATATCGG-1', 'AAAGCCTGTATGCG-1', 'AAAGGCCTGTCTAG-1','AAAGTTTGATCACG-1', 'AAAGTTTGGGGTGA-1', 'AAAGTTTGTAGAGA-1',
                       'AAAGTTTGTAGCGT-1', 'AAATCAACAATGCC-1', 'AAATCAACACCAGT-1','AAATCAACCAGGAG-1', 'AAATCAACCCTATT-1', 'AAATCAACGGAAGC-1',
                       'AAATCAACTCGCAA-1', 'AAATCATGACCACA-1', 'AAATCCCTCCACAA-1','AAATCCCTGCTATG-1', 'AAATGTTGAACGAA-1', 'AAATGTTGCCACAA-1',
                       'AAATGTTGTGGCAT-1', 'AAATTCGAAGGTTC-1', 'AAATTCGAATCACG-1','AAATTCGAGCTGAT-1',
                       'AAACATTCAACCAC-1', 'AAACATTGATCTAC-1', 'AAACATTGATCCGC-1','AAACAGTGCTTCCG-1', 'CAACCGTGTATGCG-1', 'ATACGCACTGGTAC-1',
                       'AAACGCTGATCAGT-1', 'AAACGCGGGTTCTT-1', 'AAAAGCTGTAGCCA-1','AAACTCTGTTTCTG-1']


        
class TestQCFunctions(object):
    """  Test functions for the Quality Control module of Scedar"""
    
    np.random.seed(123)
    
    mtx_df_50x40 = pd.DataFrame(np.random.randint(0,100,size=(50,40)))
    mtx_df_zeros  =  pd.DataFrame(np.zeros((50,40)))
    mtx_df_empty = pd.DataFrame()
    
    csc_50x40 = mat_to_csc(mtx_df_50x40) # self. ?
    csr_50x40 = spsp.csr.csr_matrix(mtx_df_50x40)
    
    genes = get_test_genes()
    barcodes = get_test_barcodes()
        
    def test_null_inputs(self):
        with pytest.raises(ValueError,match=r"Empty matrix found!"):
            qc=QualityControl(self.mtx_df_empty,self.genes,self.barcodes)  # change input to  []
        with pytest.raises(ValueError,match=r"Empty gene list found!"):
            qc=QualityControl(self.mtx_df_50x40,[],self.barcodes)
        with pytest.raises(ValueError,match=r"Empty barcode list found!"):
            qc=QualityControl(self.mtx_df_50x40,self.genes,[]) 
            
    def test_wrong_matrix_type(self):
        with pytest.raises(TypeError):
            qc=QualityControl(np.asarray(self.mtx_df_50x40),self.genes,self.barcodes)
    
    def test_wrong_gene_length(self):
        #with pytest.raises(ValueError,match=r"Incorrect gene list found!"):
        #    qc=QualityControl(self.mtx_df_50x40,self.genes[:-5],self.barcodes) 
        pass
    
    def test_wrong_barcode_length(self):
        pass
    

####### TEST QC_metrics ########
################################
    def test_QC_metrics_no_filter_DATAFRAME(self):
        '''test QC_metrics() and return QC_metaobj only, (no filtered data)'''    # add generate_report() testing here
        qc=QualityControl(self.mtx_df_50x40,self.genes,self.barcodes)
        QC_metaobj_50x40  = qc.QC_metrics(UMI_thresh = 1500,Features_thresh = 39,log10FeaturesPerUMI_thresh = 0.002,
                                            FeaturesPerUMI_thresh = 0.0001,mtRatio_thresh = 0.5,qc_filter=False)

        assert  QC_metaobj_50x40.shape == (len(self.mtx_df_50x40),6)
        assert  np.all(QC_metaobj_50x40.columns == ['nUMI', 'nFeatures', 'FeaturesPerUMI','log10FeaturesPerUMI', 'mtUMI','mitoRatio'])
        assert np.any(QC_metaobj_50x40.isna()) == False 
        assert sum(sum(QC_metaobj_50x40.values)) ==  107641.1346368048
        np.testing.assert_approx_equal(sum(sum(QC_metaobj_50x40.values)), 107641.13463,significant=4, err_msg='QC_metrics sum incorrect')  

    
    def test_QC_metrics_no_filter_CSC(self):

        qc=QualityControl(self.csc_50x40,self.genes,self.barcodes)
        QC_metaobj_50x40_csc  = qc.QC_metrics(qc_filter=False) # UMI_thresh = 1500,Features_thresh = 39,log10FeaturesPerUMI_thresh = 0.002,FeaturesPerUMI_thresh = 0.0001,mtRatio_thresh = 0.5,
        
        assert  QC_metaobj_50x40_csc.shape == (self.csc_50x40.shape[0],6)
        assert  np.all(QC_metaobj_50x40_csc.columns == ['nUMI', 'nFeatures', 'FeaturesPerUMI','log10FeaturesPerUMI', 'mtUMI','mitoRatio'])
        assert np.any(QC_metaobj_50x40_csc.isna()) == False 
        np.testing.assert_approx_equal(sum(sum(QC_metaobj_50x40_csc.values)), 107641.13463,significant=4, err_msg='QC_metrics sum incorrect')  

        

    ###### QC_metrics WITH filter #########
    def test_QC_metrics_with_filter_DATAFRAME(self):
        '''test QC_metrics() and return QC_metaobj AND filtered data'''   # add generate_report() testing here
        qc=QualityControl(self.mtx_df_50x40,self.genes,self.barcodes)


        fdata, fbc,fgenes, QC_metaobj_50x40  = qc.QC_metrics(UMI_thresh = 1500,Features_thresh = 39,log10FeaturesPerUMI_thresh = 0.002,
                                                        FeaturesPerUMI_thresh = 0.0001,mtRatio_thresh = 0.5, 
                                                                  qc_filter=True, remove_cell_cycle=False)
  
        assert  QC_metaobj_50x40.shape == (len(self.mtx_df_50x40),6)
        assert  np.all(QC_metaobj_50x40.columns == ['nUMI', 'nFeatures', 'FeaturesPerUMI','log10FeaturesPerUMI', 'mtUMI','mitoRatio'])
        assert np.any(QC_metaobj_50x40.isna()) == False 
        assert isinstance(fdata,spsp.csc.csc_matrix) 
        assert isinstance(fbc,list)
        assert isinstance(fgenes,list)
        assert len(fbc) == 32
        assert len(fgenes) == 40
        assert fdata.shape == (32,40)
        assert fdata.sum() == 63358
        assert len(set(fbc).difference(set(['AAACATTGAGCTAC-1','AAACATACAACCAC-1','AAACATTGATCAGC-1','AAACCGTGTATGCG-1','AAACGCACTGGTAC-1',
             'AAACGCTGACCAGT-1','AAACTTGATCCAGA-1', 'AAAGAGACGCGAGA-1', 'AAAGAGACGGCATT-1','AAAGCAGAAGCCAT-1', 'AAAGCAGATATCGG-1','AAAGCCTGTATGCG-1',
             'AAAGTTTGATCACG-1', 'AAAGTTTGGGGTGA-1', 'AAAGTTTGTAGAGA-1','AAAGTTTGTAGCGT-1', 'AAATCAACCCTATT-1', 'AAATCAACGGAAGC-1','AAATCAACTCGCAA-1',
             'AAATCCCTCCACAA-1', 'AAATCCCTGCTATG-1','AAATGTTGAACGAA-1', 'AAATGTTGCCACAA-1','AAATGTTGTGGCAT-1','AAATTCGAAGGTTC-1','AAATTCGAGCTGAT-1','AAACATTGATCTAC-1','AAACATTGATCCGC-1',
             'ATACGCACTGGTAC-1', 'AAACGCTGATCAGT-1','AAACGCGGGTTCTT-1','AAAAGCTGTAGCCA-1']))) == 0
       
        assert len(set(fgenes).difference(set(['ENSG00000243485',
                    'ENSG00000237613','ENSG00000186092','ENSG00000238009','ENSG00000239945','ENSG00000237683', 'ENSG00000239906',
                     'ENSG00000241599','ENSG00000228463','ENSG00000237094','ENSG00000235249', 'ENSG00000236601','ENSG00000236743', 'ENSG00000231709',
                     'ENSG00000239664', 'ENSG00000230021', 'ENSG00000223659','ENSG00000185097','ENSG00000235373', 'ENSG00000240618','ENSG00000229905',
                     'ENSG00000010292','ENSG00000011426','ENSG00000129055', 'ENSG00000177757', 'ENSG00000225880', 'ENSG00000230368','ENSG00000269308',
                     'ENSG00000272438','ENSG00000230699','ENSG00000210049','ENSG00000211459','ENSG00000097007', 'ENSG00000210082',
                     'ENSG00000241180', 'ENSG00000223764', 'ENSG00000187634','ENSG00000268179',
                                               'ENSG00000188976','ENSG00000187961']))) == 0

    def test_QC_metrics_with_filter_df_verbose(self):
        '''test QC_metrics() and return QC_metaobj AND filtered data'''   # add generate_report() testing here
        qc=QualityControl(self.mtx_df_50x40,self.genes,self.barcodes)
        
        fdata, fbc,fgenes, QC_metaobj_50x40  = qc.QC_metrics(UMI_thresh = 1500,
                                                             Features_thresh = 39,
                                                             log10FeaturesPerUMI_thresh = 0.002,
                                                             FeaturesPerUMI_thresh = 0.0001,
                                                             mtRatio_thresh = 0.5, 
                                                             qc_filter=True, 
                                                             remove_cell_cycle=False,
                                                             verbose=True)

############################
#### TEST QC_filter ########
############################

    def test_QC_filter_with_QC_Obj(self):
        '''test QC filter function with QC_metaobj  (call QC_metrics to get QC_metaobj  
                    and pass it to QC_filter with mtx_df)'''   
        qc=QualityControl(self.mtx_df_50x40,self.genes,self.barcodes)

        QC_metaobj_50x40 = qc.QC_metrics(qc_filter=False)
        fdata,fbc,fgenes = qc.QC_filter(QC_metaobj_50x40,remove_cell_cycle=False,UMI_thresh = 1500,Features_thresh = 39,log10FeaturesPerUMI_thresh = 0.002,
                                            FeaturesPerUMI_thresh = 0.0001,mtRatio_thresh = 0.5)
        
        assert isinstance(fdata,spsp.csc.csc_matrix)
        assert isinstance(fbc,list)
        assert isinstance(fgenes,list)
        assert np.shape(fdata) == (32,40)
        assert fdata.sum() == 63358

    def test_QC_filter_with_QC_Obj_removeCC(self):
        '''test QC filter function with QC_metaobj  (call QC_metrics to get QC_metaobj  
                    and pass it to QC_filter with mtx_df)'''   
        qc=QualityControl(self.mtx_df_50x40,self.genes,self.barcodes)

        QC_metaobj_50x40 = qc.QC_metrics(UMI_thresh = 1500,Features_thresh = 39,log10FeaturesPerUMI_thresh = 0.002,
                                            FeaturesPerUMI_thresh = 0.0001,mtRatio_thresh = 0.5,qc_filter=False)
        
        fdata, fbc,fgenes= qc.QC_filter(QC_metaobj_50x40,remove_cell_cycle=True,UMI_thresh = 1500,Features_thresh = 39,log10FeaturesPerUMI_thresh = 0.002,
                                            FeaturesPerUMI_thresh = 0.0001,mtRatio_thresh = 0.5)#,nUMI=500,nFeatures=500,FeaturesPerUMI=0.3,mtRatio=0.05)
        assert isinstance(fdata,spsp.csc.csc_matrix)
        assert isinstance(fbc,list)
        assert isinstance(fgenes,list)
        assert np.shape(fdata) == (32,38)
        assert fdata.sum() == 60257
        
        
    def test_QC_filter_without_QC_Obj(self):
        qc=QualityControl(self.mtx_df_50x40,self.genes,self.barcodes)
        fdata, fbc,fgenes = qc.QC_filter(QC_metaobj=None,remove_cell_cycle=False,UMI_thresh = 1500,Features_thresh = 39,log10FeaturesPerUMI_thresh = 0.002,
                                            FeaturesPerUMI_thresh= 0.0001,mtRatio_thresh = 0.5)
        assert isinstance(fdata,spsp.csc.csc_matrix)
        assert np.shape(fdata) == (32,40)
        assert fdata.sum() == 63358
         
    def test_QC_filter_both(self):  
        qc=QualityControl(self.mtx_df_50x40,self.genes,self.barcodes)
        
        QC_metaobj_50x40 = qc.QC_metrics(qc_filter=False)
        fdata1,bc1,genes1 =  qc.QC_filter(QC_metaobj_50x40,remove_cell_cycle=False,UMI_thresh = 1500,Features_thresh = 39,log10FeaturesPerUMI_thresh = 0.002,FeaturesPerUMI_thresh= 0.0001,mtRatio_thresh = 0.5)
        
        qc2=QualityControl(self.mtx_df_50x40,self.genes,self.barcodes) 
        fdata2, bc2, genes2 = qc2.QC_filter(QC_metaobj=None,remove_cell_cycle=False,UMI_thresh = 1500,Features_thresh = 39,log10FeaturesPerUMI_thresh = 0.002,FeaturesPerUMI_thresh= 0.0001,mtRatio_thresh = 0.5)
        #return fdata1,fdata2    #, bc2, genes2
        assert fdata1.shape == fdata2.shape
        assert np.all(bc1  ==  bc2)
        assert np.all(genes1  ==  genes2)

    def test_QC_filter_both_remove_cc(self):  
        qc=QualityControl(self.mtx_df_50x40,self.genes,self.barcodes)
        
        QC_metaobj_50x40 = qc.QC_metrics(qc_filter=False)
        fdata1,bc1,genes1 =  qc.QC_filter(QC_metaobj_50x40,remove_cell_cycle=True,UMI_thresh = 1500,Features_thresh = 39,log10FeaturesPerUMI_thresh = 0.002,
                                            FeaturesPerUMI_thresh= 0.0001,mtRatio_thresh = 0.5)
        
        qc2=QualityControl(self.mtx_df_50x40,self.genes,self.barcodes) 
        fdata2, bc2, genes2 = qc2.QC_filter(QC_metaobj=None,remove_cell_cycle=True,UMI_thresh = 1500,Features_thresh = 39,log10FeaturesPerUMI_thresh = 0.002,
                                            FeaturesPerUMI_thresh= 0.0001,mtRatio_thresh = 0.5)
        assert fdata1.shape == fdata2.shape
        assert np.all(bc1 == bc2)
        assert np.all(genes1 == genes2)


###########################################
######## Individual Filtering Tests #######
###########################################  barcodes are now returned...


    def test_QC_filter_umi(self):
            qc=QualityControl(self.mtx_df_50x40,self.genes,self.barcodes)
            fdata,fbc,fgenes = qc.QC_filter(UMI_thresh=1700)
            assert fdata.shape == (45, 40)
            
    def test_QC_filter_features(self):
            qc=QualityControl(self.mtx_df_50x40,self.genes,self.barcodes)
            fdata,fbc,fgenes = qc.QC_filter(Features_thresh=39)
            assert fdata.shape == (32, 40)

    def test_QC_filter_featuresPerUMI(self):
            qc=QualityControl(self.mtx_df_50x40,self.genes,self.barcodes)
            fdata,fbc,fgenes = qc.QC_filter(FeaturesPerUMI_thresh=.02)
            assert fdata.shape == (30, 40)

    def test_QC_filter_log10featuresPerUMI(self):
            qc=QualityControl(self.mtx_df_50x40,self.genes,self.barcodes)
            fdata,fbc,fgenes = qc.QC_filter(log10FeaturesPerUMI_thresh=.48)
            assert fdata.shape == (43, 40)

    def test_QC_filter_mtRatio(self):
            qc=QualityControl(self.mtx_df_50x40,self.genes,self.barcodes)
            fdata,fbc,fgenes = qc.QC_filter(mtRatio_thresh=.1)
            assert fdata.shape == (43,40)

    def test_QC_filter_allFilters(self):
            qc=QualityControl(self.mtx_df_50x40,self.genes,self.barcodes)
            fdata,fbc,fgenes = qc.QC_filter(UMI_thresh=1700,Features_thresh=39,FeaturesPerUMI_thresh=.02,
                                          log10FeaturesPerUMI_thresh=.48,mtRatio_thresh=.1)
            assert fdata.shape == (14, 40)
            
    def test_QC_filter_allFilters_compare(self):
            qc=QualityControl(self.mtx_df_50x40,self.genes,self.barcodes)
            
            d0,a,b = qc.QC_filter(UMI_thresh=1700)            
            d1,a,b = qc.QC_filter(Features_thresh=39)
            d2,a,b = qc.QC_filter(FeaturesPerUMI_thresh=.02)
            d3,a,b = qc.QC_filter(log10FeaturesPerUMI_thresh=.48)
            d4,fbc4,fgenes4 = qc.QC_filter(mtRatio_thresh=.1)
            
            qc=QualityControl(self.mtx_df_50x40,self.genes,self.barcodes)
            d_all,fbc_all,fgenes_all = qc.QC_filter(UMI_thresh=1700,Features_thresh=39,FeaturesPerUMI_thresh=.02,
                                          log10FeaturesPerUMI_thresh=.48,mtRatio_thresh=.1)

            assert (d0.shape[0],d1.shape[0],d2.shape[0],d3.shape[0],d4.shape[0],d_all.shape[0]) == (45, 30, 17, 17, 14, 14)
            assert np.all(fgenes4 == fgenes_all)
            assert np.all(fbc4 == fbc_all)
            assert np.sum(d4) ==  np.sum(d_all)
            
    def test_log10FeaturesPerUMI_thresh(self):
        qc=QualityControl(self.mtx_df_50x40,self.genes,self.barcodes)
        fdata1,fbc1,fgenes1 = qc.QC_filter(log10FeaturesPerUMI_thresh=.48)
        
        qc=QualityControl(self.mtx_df_50x40,self.genes,self.barcodes)
        fdata2,fbc2,fgenes2,QC_metaobj_50x40 = qc.QC_metrics(qc_filter=True,log10FeaturesPerUMI_thresh=.48)
        
        assert fdata1.shape == fdata2.shape
        #return fdata1.shape,fdata2.shape

    def test_metrics_filter_no_args(self):
        qc=QualityControl(self.mtx_df_50x40,self.genes,self.barcodes)
        with pytest.raises(ValueError,match=r"""Must pass at least one threshold arguement when qc_filter = True. If you don't want
                            any filtering done, set qc_filter = False."""):
            fdata,fbc,fgenes,QC_metaobj_50x40 = qc.QC_metrics(qc_filter=True)
            
##### test threshold value error #######  
########################################

# with QC_metrics
    def test_metrics_umi_threshold_error(self):
        qc=QualityControl(self.mtx_df_50x40,self.genes,self.barcodes)
        with pytest.raises(ValueError,match=r"UMI threshold too high, all samples would be removed."):
            fdata,fbc,fgenes, QC_metaobj_50x40 = qc.QC_metrics(qc_filter=True,remove_cell_cycle=True,
                                                           UMI_thresh = 3000,Features_thresh = 39,
                                                           log10FeaturesPerUMI_thresh = 0.002,
                                                           FeaturesPerUMI_thresh = 0.0001,mtRatio_thresh = 0.5)
    def test_metrics_features_threshold_error(self):
        qc=QualityControl(self.mtx_df_50x40,self.genes,self.barcodes)
        with pytest.raises(ValueError,match=r"Feature threshold too high, all samples would be removed."):
            fdata,fbc,fgenes, QC_metaobj_50x40 = qc.QC_metrics(qc_filter=True,remove_cell_cycle=True,
                                                           UMI_thresh = 1500,Features_thresh = 50,
                                                           log10FeaturesPerUMI_thresh = 0.002,
                                                           FeaturesPerUMI_thresh = 0.0001,mtRatio_thresh = 0.5)
        
    def test_metrics_featuresPerUMI_threshold_error(self):
        qc=QualityControl(self.mtx_df_50x40,self.genes,self.barcodes)
        with pytest.raises(ValueError,match=r"Features per UMI threshold too high, all samples would be removed."):
            fdata,fbc,fgenes, QC_metaobj_50x40 = qc.QC_metrics(qc_filter=True,
                                                           remove_cell_cycle=True,
                                                           UMI_thresh = 1500,Features_thresh = 39,
                                                           log10FeaturesPerUMI_thresh = 0.002,
                                                           FeaturesPerUMI_thresh = 50,mtRatio_thresh = 0.5)
        
    def test_metrics_1og10featuresPerUMI_threshold_error(self):
        qc=QualityControl(self.mtx_df_50x40,self.genes,self.barcodes)
        with pytest.raises(ValueError,match=r"log10 Feature per UMI threshold too high, all samples would be removed."):
            fdata,fbc,fgenes, QC_metaobj_50x40 = qc.QC_metrics(qc_filter=True,
                                                           remove_cell_cycle=True,
                                                           UMI_thresh = 1500,Features_thresh = 39,
                                                           log10FeaturesPerUMI_thresh = 0.9,
                                                           FeaturesPerUMI_thresh = 0.0001,mtRatio_thresh = 0.5)
        
    def test_metrics_mtRatio_threshold_error(self):
        qc=QualityControl(self.mtx_df_50x40,self.genes,self.barcodes)
        with pytest.raises(ValueError,match=r"MT ratio threshold too low, all samples would be removed."):
            fdata,fbc,fgenes, QC_metaobj_50x40 = qc.QC_metrics(qc_filter=True,
                                                           remove_cell_cycle=True,
                                                           UMI_thresh = 1500,Features_thresh = .9,
                                                           log10FeaturesPerUMI_thresh = 0.002,
                                                           FeaturesPerUMI_thresh = 0.0001,mtRatio_thresh = .000001)

# filter threshold warnings        
    def test_QC_filter_umi_threshold_error(self):
        qc=QualityControl(self.mtx_df_50x40,self.genes,self.barcodes)
        with pytest.raises(ValueError,match=r"UMI threshold too high, all samples would be removed."):
            fdata,fbc,fgenes = qc.QC_filter(UMI_thresh=3000)
            
    def test_QC_filter_features_threshold_error(self):
        qc=QualityControl(self.mtx_df_50x40,self.genes,self.barcodes)
        with pytest.raises(ValueError,match=r"Feature threshold too high, all samples would be removed."):
            fdata,fbc,fgenes = qc.QC_filter(Features_thresh=50)  
                
    def test_QC_filter_featuresPerUMI_threshold_error(self):
        qc=QualityControl(self.mtx_df_50x40,self.genes,self.barcodes)
        with pytest.raises(ValueError,match=r"Features per UMI threshold too high, all samples would be removed."):
            fdata,fbc,fgenes = qc.QC_filter(FeaturesPerUMI_thresh=.2)

    def test_QC_filter_log10featuresPerUMI_threshold_error(self):
        qc=QualityControl(self.mtx_df_50x40,self.genes,self.barcodes)
        with pytest.raises(ValueError,match=r"log10 Feature per UMI threshold too high, all samples would be removed."):
            fdata,fbc,fgenes = qc.QC_filter(log10FeaturesPerUMI_thresh=.9)

    def test_QC_filter_mtRatio_threshold_error(self):   
        qc=QualityControl(self.mtx_df_50x40,self.genes,self.barcodes)
        with pytest.raises(ValueError,match=r"MT ratio threshold too low, all samples would be removed."):
            fdata,fbc,fgenes = qc.QC_filter(mtRatio_thresh=.000001)
            
            
###  test threshold warning WITHOUT qc_metaobj (with QC_filter)
    '''
    def test_filter_umi_threshold_error(self):
        qc=QualityControl(self.mtx_df_50x40,self.genes,self.barcodes)
        with pytest.raises(ValueError,match=r"UMI threshold too high, all samples would be removed."):
            fdata,fbc,fgenes= qc.QC_filter(remove_cell_cycle=True,
                                                           UMI_thresh = 3000,Features_thresh = 39,
                                                           log10FeaturesPerUMI_thresh = 0.002,
                                                           FeaturesPerUMI_thresh = 0.0001,mtRatio_thresh = 0.5)
    def test_filter_features_threshold_error(self):
        qc=QualityControl(self.mtx_df_50x40,self.genes,self.barcodes)
        with pytest.raises(ValueError,match=r"Feature threshold too high, all samples would be removed."):
            fdata,fbc,fgenes= qc.QC_filter(remove_cell_cycle=True,
                                                           UMI_thresh = 1500,Features_thresh = 50,
                                                           log10FeaturesPerUMI_thresh = 0.002,
                                                           FeaturesPerUMI_thresh = 0.0001,mtRatio_thresh = 0.5)
        
    def test_filter_featuresPerUMI_threshold_error(self):
        qc=QualityControl(self.mtx_df_50x40,self.genes,self.barcodes)
        with pytest.raises(ValueError,match=r"Features per UMI threshold too high, all samples would be removed."):
            fdata,fbc,fgenes= qc.QC_metrics(remove_cell_cycle=True,
                                                           UMI_thresh = 1500,Features_thresh = 39,
                                                           log10FeaturesPerUMI_thresh = 0.002,
                                                           FeaturesPerUMI_thresh = 50,mtRatio_thresh = 0.5)
        
    def test_filter_1og10featuresPerUMI_threshold_error(self):
        qc=QualityControl(self.mtx_df_50x40,self.genes,self.barcodes)
        with pytest.raises(ValueError,match=r"log10 Feature per UMI threshold too high, all samples would be removed."):
            fdata,fbc,fgenes = qc.QC_filter(remove_cell_cycle=True,
                                                           UMI_thresh = 1500,Features_thresh = 39,
                                                           log10FeaturesPerUMI_thresh = 0.9,
                                                           FeaturesPerUMI_thresh = 0.0001,mtRatio_thresh = 0.5)
        
    def test_filter_mtRatio_threshold_error(self):
        qc=QualityControl(self.mtx_df_50x40,self.genes,self.barcodes)
        with pytest.raises(ValueError,match=r"MT ratio threshold too low, all samples would be removed."):
            fdata,fbc,fgenes = qc.QC_filter(remove_cell_cycle=True,
                                                           UMI_thresh = 1500,Features_thresh = .9,
                                                           log10FeaturesPerUMI_thresh = 0.002,
                                                           FeaturesPerUMI_thresh = 0.0001,mtRatio_thresh = .000001)
    '''
        
        
        
    def test_QC_filter_umi_threshold_error(self):
        qc=QualityControl(self.mtx_df_50x40,self.genes,self.barcodes)
        with pytest.raises(ValueError,match=r"UMI threshold too high, all samples would be removed."):
            fdata,fbc,fgenes = qc.QC_filter(UMI_thresh=3000)
            
    def test_QC_filter_features_threshold_error(self):
        qc=QualityControl(self.mtx_df_50x40,self.genes,self.barcodes)
        with pytest.raises(ValueError,match=r"Feature threshold too high, all samples would be removed."):
            fdata,fbc,fgenes = qc.QC_filter(Features_thresh=50)  
                
    def test_QC_filter_featuresPerUMI_threshold_error(self):
        qc=QualityControl(self.mtx_df_50x40,self.genes,self.barcodes)
        with pytest.raises(ValueError,match=r"Features per UMI threshold too high, all samples would be removed."):
            fdata,fbc,fgenes = qc.QC_filter(FeaturesPerUMI_thresh=.2)

    def test_QC_filter_log10featuresPerUMI_threshold_error(self):
        qc=QualityControl(self.mtx_df_50x40,self.genes,self.barcodes)
        with pytest.raises(ValueError,match=r"log10 Feature per UMI threshold too high, all samples would be removed."):
            fdata,fbc,fgenes = qc.QC_filter(log10FeaturesPerUMI_thresh=.9)

    def test_QC_filter_mtRatio_threshold_error(self):   
        qc=QualityControl(self.mtx_df_50x40,self.genes,self.barcodes)
        with pytest.raises(ValueError,match=r"MT ratio threshold too low, all samples would be removed."):
            fdata,fbc,fgenes = qc.QC_filter(mtRatio_thresh=.000001)

            
            
            
            
###### test helper functions ##########
    def test_get_mt_idx(self):
        qc=QualityControl(self.mtx_df_50x40,self.genes,self.barcodes)
        mt_idx = qc.get_mt_idx(self.genes)
        mt_idx_assarray = qc.get_mt_idx(np.array(self.genes))
        assert isinstance(mt_idx,list)
        assert mt_idx_assarray == [30, 31, 33]
        
    def test_get_cc_idx(self):
        qc=QualityControl(self.mtx_df_50x40,self.genes,self.barcodes)
        cc_idx = qc.get_cc_idx(self.genes)
        cc_idx_assarray = qc.get_cc_idx(np.array(self.genes))
        assert isinstance(cc_idx_assarray,list)
        assert cc_idx_assarray == [32, 23]
        
        
    def test_mat_to_csc(self):
        qc=QualityControl(self.mtx_df_50x40,self.genes,self.barcodes)
        
        mat = qc.mat_to_csc(self.mtx_df_50x40)
        assert isinstance(mat,scipy.sparse.csc.csc_matrix)
        assert mat.shape == (50, 40)
        
        mat = qc.mat_to_csc(self.csr_50x40)
        assert isinstance(mat,scipy.sparse.csc.csc_matrix)
        assert mat.shape == (50, 40)
        
        mat = qc.mat_to_csc(spsp.coo.coo_matrix(self.mtx_df_50x40))
        assert isinstance(mat,scipy.sparse.csc.csc_matrix)
        assert mat.shape == (50, 40)
            
    def test_get_mito_genes_human(self): 
        qc=QualityControl(self.mtx_df_50x40,self.genes,self.barcodes)
        mt_genes =  qc.get_mito_genes('human')
        
        assert isinstance(mt_genes,list)
        assert np.shape(mt_genes) == (37,)
        assert mt_genes[0] == 'ENSG00000210049'
        assert mt_genes[-1] ==  'ENSG00000210196'
        
    def test_get_mito_genes_mouse(self): 
        qc=QualityControl(self.mtx_df_50x40,self.genes,self.barcodes)
        mt_genes =  qc.get_mito_genes('mouse')
        
        assert isinstance(mt_genes,list)
        assert np.shape(mt_genes) == (37,)
        assert mt_genes[0] == 'ENSMUSG00000064336'
        assert mt_genes[-1] ==  'ENSMUSG00000064372'


    def test_get_cell_cycle_genes_human(self): 
        qc=QualityControl(self.mtx_df_50x40,self.genes,self.barcodes)
        cc_genes =  qc.get_cell_cycle_genes('human')
        assert isinstance(cc_genes,list)
        assert np.shape(cc_genes) == (125,)
        assert cc_genes[0] == 'ENSG00000097007'
        assert cc_genes[-1] ==  'ENSG00000116809'


    def test_get_cell_cycle_genes_mouse(self):   
        qc=QualityControl(self.mtx_df_50x40,self.genes,self.barcodes)
        cc_genes =  qc.get_cell_cycle_genes('mouse')
        assert isinstance(cc_genes,list)
        assert np.shape(cc_genes) == (125,)
        assert cc_genes[0] == 'ENSMUSG00000026842'
        assert cc_genes[-1] ==  'ENSMUSG00000006215'

