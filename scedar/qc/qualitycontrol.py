import numpy as np 
import pandas as pd
import os  
import sys
import warnings
import scipy
import scipy.sparse as spsp
#import collections


class QualityControl(object):
    """
    Single Cell Quality Control
    
    Args
    ----
    
    mtx_df : Pandas DataFrame or Scipy sparse matrix
        Single cell count matrix with samples as rows and features as columns.
    genes : list
        Genes corresponding to count matrix
    barcodes : list
        Barcode IDs corresponding to count matrix
        
    
    Attributes
    ----------
    
    _mtx_df : Scipy CSC matrix
        Single cell count matrix with samples as rows and features as columns.
    _genes : list
        Genes corresponding to count matrix
    _barcodes : list
        Barcode IDs corresponding to count matrix   
        
    """
    
    def __init__(self, mtx_df=None,genes=None, barcodes=None): # ,nprocs=1,verbose=False):
        super().__init__()
        self._mtx_df  = mtx_df
        self._genes = genes
        self._barcodes = barcodes 
        self._init_samples = self._mtx_df.shape[0]
        
        if self._mtx_df is not None:
                if not isinstance(self._mtx_df,(pd.core.frame.DataFrame,spsp.coo.coo_matrix
                                                 ,spsp.csr.csr_matrix,spsp.csc.csc_matrix)):           
                    raise TypeError(f"""Input Matrix must be of Type 'Pandas.core.frame.DataFrame',
                    'scipy.sparse.coo.coo_matrix' or 'scipy.sparse.csc.csc_matrix' but was Type: {type(self._mtx_df)}""")
                    
                elif not isinstance(self._mtx_df,spsp.csc.csc_matrix):
                    self._mtx_df = self.mat_to_csc(self._mtx_df)
        else: 
            raise ValueError("No count matrix entered") 
                #if not (len(self._barcodes) == self._mtx_df.shape[0]): 
                #    raise ValueError('Barcodes do not match length of matrix on axis 0 (columns). Make sure your matrix is in the form "features X barcodes".')
                #if not (len(self._genes) == self._mtx_df.shape[1]):  
                #    raise ValueError('Features do not match length of matrix on axis 1 (rows). Make sure your matrix is in the form "features X barcodes".')
        
        #print(f'Shape of count matrix is {self._mtx_df.shape}')

        if not self._mtx_df.shape[0]: raise ValueError('Empty matrix found!')
        if not len(self._genes): raise ValueError('Empty gene list found!')
        if not len(self._barcodes): raise ValueError('Empty barcode list found!')
            
        #if self._mtx_df is not None and self._genes is None and self._barcodes is None:
        #    self._genes =  self._mtx_df.columns.to_list()
        #    self._barcodes = self._mtx_df.index.to_list()
        
    def QC_metrics(self,qc_filter=False,remove_cell_cycle=False,report=False,UMI_thresh  = 0,Features_thresh = 0,
                   log10FeaturesPerUMI_thresh = 0.0,FeaturesPerUMI_thresh = 0.0,mtRatio_thresh = 1.0, verbose=False):
        """
        Calculate quality control metrics, and optionally filter count matrix and
        generate a quality control report.
        
        Args
        ----
        
        qc_filter : bool
        If True, a filtered count matrix will be returned 
        
        """
        if qc_filter and not np.any([remove_cell_cycle,report,UMI_thresh,Features_thresh,
                   log10FeaturesPerUMI_thresh,FeaturesPerUMI_thresh])  and mtRatio_thresh is not 1.0:
            raise ValueError("""Must pass at least one threshold arguement when qc_filter = True. If you don't want
                            any filtering done, set qc_filter = False.""")
            
        
        idx = self.get_mt_idx(self._genes)
            
        QC_metaobj =  pd.DataFrame(index=self._barcodes).rename_axis(index="Barcodes")
        
        QC_metaobj['nUMI'] = self._mtx_df.sum(axis=1) 
        QC_metaobj['nFeatures'] = self._mtx_df.astype(bool).sum(axis=1).astype(int)
        QC_metaobj['FeaturesPerUMI'] = np.divide(QC_metaobj['nFeatures'], QC_metaobj['nUMI'])
        QC_metaobj['log10FeaturesPerUMI'] = np.divide(np.log10(QC_metaobj['nFeatures']), np.log10(QC_metaobj['nUMI']))
        QC_metaobj['mtUMI'] = self._mtx_df[:,idx].sum(axis=1).astype(int)
        QC_metaobj['mitoRatio'] = QC_metaobj['mtUMI']/QC_metaobj['nUMI']       

        '''if report:  
            gene_exp=  pd.DataFrame(self._mtx_df.sum(axis=0).T,index=self._genes)/ self._mtx_df.sum()
            sdict  = {'Total Cells' : self._mtx_df.shape[0],
                  'Total Features'  : self._mtx_df.shape[1],
                  'Total UMIs'  : int(self._mtx_df.sum()),
                  'Unique Features' : int(self._mtx_df.astype(bool).sum(axis=0).astype(bool).sum()),
                  'Median Features per cell' : int(QC_metaobj['nFeatures'].median()),
                  'Median UMIs per cell' : int(QC_metaobj['nUMI'].median()),
                  'Median MT Ratio':round(QC_metaobj['mitoRatio'].median(),6), # rounds to nearest  even num, but close enough for 6 decimals out, to use f-strings, do  {num:.2f}
                  'MT genes detected' : len(idx)} 
            
               #'Sparsity':  self._mtx_df.getnnz() / np.prod(self._mtx_df.shape)}
            fdict = {'UMI Cutoff' : UMI_thresh,
                  'Feature cutoff' : Features_thresh,
                  'Features per UMI cutoff' : FeaturesPerUMI_thresh,
                  'MT Ratio cutoff' : mtRatio_thresh,
                  #  get filtered_data
                    'Samples Removed': 44,
                   'Percent Samples Removed': .22}
            #if qc_filter: 
            #     filtered_data[]
            #    generate_report(sdict+fdict,QC_metaobj,gene_exp) 
            #else: generate_report(sdict,QC_metaobj,gene_exp)
            self.generate_report(sdict,QC_metaobj,gene_exp)'''
            
        if qc_filter:  
            filtered_data, barcodes, genes  = self.QC_filter(QC_metaobj=QC_metaobj,
                                            remove_cell_cycle=remove_cell_cycle,
                                            UMI_thresh=UMI_thresh, 
                                            Features_thresh=Features_thresh,
                                            log10FeaturesPerUMI_thresh=log10FeaturesPerUMI_thresh,
                                            FeaturesPerUMI_thresh=FeaturesPerUMI_thresh,
                                            mtRatio_thresh=mtRatio_thresh,verbose=verbose)
            
            return  filtered_data, barcodes, genes, QC_metaobj   #  CHANGE ORDER ? 
        else: 
            return QC_metaobj
        
    #def foo(*,arg0="default0", arg1="default1", arg2="default2"):
    #    pass
    
    
    def QC_filter(self,QC_metaobj=None,remove_cell_cycle=False,UMI_thresh  = 0,Features_thresh = 0,
                   log10FeaturesPerUMI_thresh = 0.0,FeaturesPerUMI_thresh = 0.0,mtRatio_thresh = 1.0,verbose=False):
        """
        Function for filtering count matrix.
        
        Args
        ----
        
        QC_metaobj: qc metaobject DataFrame.
        
        Default is None. If no qc meta object is passed, the function will generate qc metrics and then filter.
                
            Whether or not a QC metaobj is being passed. QC metaobj is generated 
            with the QC_metrics() function.
        remove_cell_cycle: bool
            Whether or not to remove cell cycle genes.
        UMI_thresh: int
            Threshold to filter count matrix based on UMI count per cell.
        Features_thresh: int
            Threshold to filter count matrix based on number of unique features detected per cell.
        log10FeaturesPerUMI_thresh: float
            Same as Features_thresh but in log10 space.
        FeaturesPerUMI_thresh: float
            Threshold to filter count matrix based on number of features per UMI detected per cell.
        mtRatio_thresh: float
            Threshold to filter count matrix based on percentage of mitochondrial UMI per cell.        
        """

        if QC_metaobj is not None:   
            if UMI_thresh and not sum(QC_metaobj['nUMI'] > UMI_thresh): raise ValueError("UMI threshold too high, all samples would be removed.")
            if Features_thresh and not sum(QC_metaobj['nFeatures'] > Features_thresh): raise ValueError("Feature threshold too high, all samples would be removed.")
            if FeaturesPerUMI_thresh and not sum(QC_metaobj['FeaturesPerUMI'] > FeaturesPerUMI_thresh): raise ValueError("Features per UMI threshold too high, all samples would be removed.")
            if log10FeaturesPerUMI_thresh and not sum(QC_metaobj['log10FeaturesPerUMI'] > log10FeaturesPerUMI_thresh): raise ValueError("log10 Features per UMI threshold too high, all samples would be removed.") 
            if mtRatio_thresh and not sum(QC_metaobj['mitoRatio'] < mtRatio_thresh): raise ValueError("MT ratio threshold too low, all samples would be removed.")            
            
            mask_all =(QC_metaobj['nUMI'] > UMI_thresh).values & \
                           (QC_metaobj['nFeatures'] > Features_thresh).values & \
                           (QC_metaobj['FeaturesPerUMI'] > FeaturesPerUMI_thresh).values &  \
                              (QC_metaobj['log10FeaturesPerUMI']  >  log10FeaturesPerUMI_thresh).values & \
                           (QC_metaobj['mitoRatio'] < mtRatio_thresh).values
            
            
            if remove_cell_cycle:
                idx_cc = self.get_cc_idx(self._genes)
                to_keep = list(set(range(self._mtx_df.shape[1]))-set(idx_cc))    
                self._mtx_df = self._mtx_df[:,to_keep]
                #self._genes = list(set(genes) - set(mt_genes))  #  self._genes[to_keep] see if this is faster
                self._genes =   [self._genes[i] for i in to_keep]

            '''
            umi_sum = sum(QC_metaobj['nUMI'] > UMI_thresh)
            ftr_sum = sum(QC_metaobj['nFeatures'] > Features_thresh)
            ftr_per_umi_sum = sum(QC_metaobj['FeaturesPerUMI'] > FeaturesPerUMI_thresh)
            log10_ftr_per_umi_sum = sum(QC_metaobj['log10FeaturesPerUMI']  >  log10FeaturesPerUMI_thresh)
            mt_sum = sum(QC_metaobj['mitoRatio'] < mtRatio_thresh)
            
            print(f'UMI_thresh: {self._init_samples-umi_sum} ')
            print(f'ftr_thresh: {self._init_samples-ftr_sum} ')
            print(f'ftr_per_umi_thresh: {self._init_samples-ftr_per_umi_sum} ')
            print(f'log10_ftr_per_umi_thresh: {self._init_samples-log10_ftr_per_umi_sum} ')
            print(f'mt: {self._init_samples-mt_sum} ')
            '''
            print(f'Total Samples removed: { self._init_samples-sum(mask_all)} ')
            
            self._mtx_df =  self._mtx_df[mask_all]
            self._barcodes = list(np.array(self._barcodes)[mask_all])

            return self._mtx_df, self._barcodes, self._genes

        else: # compute QC_metaobj stats and filter bycutoffs now
            print('here')
            init_samples, init_genes = self._mtx_df.shape
            self._barcodes = np.array(self._barcodes)
            
            if UMI_thresh:   #   TEST FOR ISINSTANCE AND VERBOSE
                
                if not isinstance(UMI_thresh,int): raise ValueError("UMI threshold must be an integer.")
                mask_umi = np.ravel((self._mtx_df.sum(axis=1) > UMI_thresh).tolist())
                if not sum(mask_umi): raise ValueError("UMI threshold too high, all samples would be removed.")
                #self._mtx_df = self._mtx_df[mask_umi,:]
                #self._barcodes = self._barcodes[mask_umi]
                if verbose: print(f'{init_samples  - self._mtx_df.shape[0]} samples removed after applying UMI_thresh of {UMI_thresh}')
            else: mask_umi = np.ones(init_samples).astype(bool)

            if Features_thresh:
                #print(self._mtx_df.astype(bool).sum(axis=1))
                if not isinstance(Features_thresh,int): raise ValueError("Feature threshold must be an integer.") 
                current_shape = self._mtx_df.shape[0]
                mask_ftr= np.ravel((self._mtx_df.astype(bool).sum(axis=1)  > Features_thresh).tolist())
                #print(mask_ftr)
                if not sum(mask_ftr): raise ValueError("Feature threshold too high, all samples would be removed.")
                if verbose: print(f'{current_shape  - self._mtx_df.shape[0]} samples removed after applying Features_thresh of {Features_thresh}')
            else: mask_ftr = np.ones(init_samples).astype(bool)
            
            if FeaturesPerUMI_thresh:
                if not isinstance(FeaturesPerUMI_thresh,(int,float)):raise ValueError("FeaturePerUMI threshold must be an integer or float."); #sys.exit(1)
                current_shape = self._mtx_df.shape[0]
                ftr_per_umi_mask =  np.ravel((np.divide(self._mtx_df.astype(bool).sum(axis=1).astype(int) ,self._mtx_df.sum(axis=1)) > FeaturesPerUMI_thresh).tolist())                
                if not sum(ftr_per_umi_mask): raise ValueError("Features per UMI threshold too high, all samples would be removed.")
                if verbose: print(f'{current_shape  - self._mtx_df.shape[0]} samples removed after applying featurePerUMI_thresh of {FeaturesPerUMI_thresh}')
            else: ftr_per_umi_mask = np.ones(init_samples).astype(bool)
            
            if log10FeaturesPerUMI_thresh:
                if not isinstance(log10FeaturesPerUMI_thresh,(int,float)): raise ValueError("log10featurePerUMI threshold must be an integer or float."); #sys.exit(1)
                current_shape = self._mtx_df.shape[0]
                # calculate division first then log10, if theres a difference.
                log10_ftr_per_umi_mask =  np.ravel((np.log10(self._mtx_df.astype(bool).sum(axis=1)) / np.log10(self._mtx_df.sum(axis=1).astype(int)) > log10FeaturesPerUMI_thresh).tolist())
                if not sum(log10_ftr_per_umi_mask): raise ValueError("log10 Feature per UMI threshold too high, all samples would be removed.")
                if verbose: print(f'{current_shape  - self._mtx_df.shape[0]} samples removed after applying log10featurePerUMI_thresh of {log10FeaturesPerUMI_thresh}')
            else: log10_ftr_per_umi_mask = np.ones(init_samples).astype(bool)
                
            if not (mtRatio_thresh == 1.0):
                if not isinstance(mtRatio_thresh,float) or (mtRatio_thresh) < 0 or (mtRatio_thresh > 1): raise ValueError("mtRatio threshold must be a float between 0 and 1."); #sys.exit(1)
                current_shape = self._mtx_df.shape[0]
                mt_idx =  self.get_mt_idx(self._genes)
                mt_mask = np.ravel((self._mtx_df[:,mt_idx].sum(axis=1).astype(int) /  self._mtx_df.sum(axis=1).astype(int) < mtRatio_thresh).tolist())
                if not sum(mt_mask): raise ValueError("MT ratio threshold too low, all samples would be removed.")
                #self._mtx_df =  self._mtx_df[mt_mask]
                #self._barcodes = self._barcodes[mt_mask]
                if verbose: print(f'{current_shape  - self._mtx_df.shape[0]} samples removed after applying mtRatio_thresh of {mtRatio_thresh}')
            else: mt_mask = np.ones(init_samples).astype(bool)
            
            if remove_cell_cycle:
                idx_cc = self.get_cc_idx(self._genes)
                to_keep = list(set(range(self._mtx_df.shape[1]))-set(idx_cc))    
                self._mtx_df = self._mtx_df[:,to_keep]  #self._genes = list(set(genes) - set(mt_genes))  #  self._genes[to_keep]
                self._genes =   [self._genes[i] for i in to_keep]
                if verbose: print(f'{init_genes  - self._mtx_df.shape[1]} cell cycle genes found and removed')
            
            if verbose: print(f'{init_samples - self._mtx_df.shape[0]} samples and {init_genes - self._mtx_df.shape[1]} dropped ({np.round(1 - (self._mtx_df.shape[0]+self._mtx_df.shape[1])/(init_genes+init_samples),3)*100}% of matrix).')
            '''umi_sum = sum(mask_umi)
            ftr_sum = sum(mask_ftr)
            ftr_per_umi_sum = sum(ftr_per_umi_mask)
            log10_ftr_per_umi_sum = sum(log10_ftr_per_umi_mask)
            mt_sum = sum(mt_mask)
            
            print(f'UMI_thresh: {self._init_samples-umi_sum} ')
            print(f'ftr_thresh: {self._init_samples-ftr_sum} ')
            print(f'ftr_per_umi_thresh: {self._init_samples-ftr_per_umi_sum} ')
            print(f'log10_ftr_per_umi_thresh: {self._init_samples-log10_ftr_per_umi_sum} ')
            print(f'mt: {self._init_samples-mt_sum} ')
            '''
            mask_all = mask_umi & mask_ftr & ftr_per_umi_mask & log10_ftr_per_umi_mask  & mt_mask
            self._mtx_df =  self._mtx_df[mask_all]
            self._barcodes = self._barcodes[mask_all]
            
            print(f'Total Samples removed: {self._init_samples-self._mtx_df.shape[0]}')
        
            return self._mtx_df, list(self._barcodes), self._genes

        
    def mat_to_csc(self, mat, verbose=0):
        #if mtx_df.shape[0] > mtx_df.shape[1]: return sparse.csr_matrix(df.values) # more cells than genes, convert to csr
        #else: return sparse.csc_matrix(df.values) # more genes than cells, convert to csc  
        if isinstance(mat,pd.core.frame.DataFrame):
            if verbose: print('DataFrame converted to CSC matrix.')
            return spsp.csc_matrix(mat.values)
        elif  isinstance(mat,spsp.csr.csr_matrix):
            if verbose: print('CSR matrix converted to CSC matrix.')
            return spsp.csc_matrix(mat)
        elif isinstance(mat,spsp.coo.coo_matrix):
            if verbose: print('COO matrix converted to CSC matrix.')
            return spsp.csc_matrix(mat) #mat.tocsc()

    #@staticmethod
    def get_mito_genes(self,organism='human'):
        if organism=='human':
            mt_genes =  ['ENSG00000210049','ENSG00000211459','ENSG00000210077','ENSG00000210082',
                             'ENSG00000209082','ENSG00000198888', 'ENSG00000210100', 'ENSG00000210107', 'ENSG00000210112',
                             'ENSG00000198763', 'ENSG00000210117','ENSG00000210127','ENSG00000210135','ENSG00000210140',
                             'ENSG00000210144','ENSG00000198804','ENSG00000210151','ENSG00000210154','ENSG00000198712',
                            'ENSG00000210156','ENSG00000228253','ENSG00000198899','ENSG00000198938','ENSG00000210164',
                             'ENSG00000198840','ENSG00000210174','ENSG00000212907','ENSG00000198886','ENSG00000210176',
                             'ENSG00000210184','ENSG00000210191','ENSG00000198786','ENSG00000198695', 'ENSG00000210194',
                             'ENSG00000198727', 'ENSG00000210195','ENSG00000210196']
        elif organism=='mouse':
            mt_genes = ['ENSMUSG00000064336', 'ENSMUSG00000064337', 'ENSMUSG00000064338',
                       'ENSMUSG00000064339', 'ENSMUSG00000064340', 'ENSMUSG00000064341','ENSMUSG00000064342', 'ENSMUSG00000064343', 'ENSMUSG00000064344',
                       'ENSMUSG00000064345', 'ENSMUSG00000064346', 'ENSMUSG00000064347','ENSMUSG00000064348', 'ENSMUSG00000064349', 'ENSMUSG00000064350',
                       'ENSMUSG00000064351', 'ENSMUSG00000064352', 'ENSMUSG00000064353','ENSMUSG00000064354', 'ENSMUSG00000064355', 'ENSMUSG00000064356',
                       'ENSMUSG00000064357', 'ENSMUSG00000064358', 'ENSMUSG00000064359','ENSMUSG00000064360', 'ENSMUSG00000064361', 'ENSMUSG00000065947',
                       'ENSMUSG00000064363', 'ENSMUSG00000064364', 'ENSMUSG00000064365','ENSMUSG00000064366', 'ENSMUSG00000064367', 'ENSMUSG00000064368',
                       'ENSMUSG00000064369', 'ENSMUSG00000064370', 'ENSMUSG00000064371','ENSMUSG00000064372']
        return mt_genes

    #@staticmethod
    def get_cell_cycle_genes(self,organism='human'):
        #cell_cycle_genes = pd.read_csv('human_cell_cycle_genes.csv',sep=',')
        #return cell_cycle_genes['geneID'].values
        if organism=='human':
            cc_genes = ['ENSG00000097007','ENSG00000256211','ENSG00000153107','ENSG00000164162','ENSG00000141552', 'ENSG00000129055','ENSG00000176248',
             'ENSG00000053900', 'ENSG00000089053', 'ENSG00000196510', 'ENSG00000149311', 'ENSG00000175054', 'ENSG00000169679',
             'ENSG00000156970','ENSG00000154473', 'ENSG00000133101', 'ENSG00000145386', 'ENSG00000134057','ENSG00000157456',
             'ENSG00000147082', 'ENSG00000110092', 'ENSG00000118971', 'ENSG00000112576','ENSG00000105173', 'ENSG00000175305',
             'ENSG00000134480','ENSG00000079335','ENSG00000081377','ENSG00000130177','ENSG00000117399', 'ENSG00000094880',
             'ENSG00000164045', 'ENSG00000101224', 'ENSG00000158402', 'ENSG00000176386', 'ENSG00000004897', 'ENSG00000093009',
             'ENSG00000094804', 'ENSG00000097046', 'ENSG00000170312','ENSG00000123374','ENSG00000135446','ENSG00000105810',
             'ENSG00000134058', 'ENSG00000124762', 'ENSG00000111276','ENSG00000129757', 'ENSG00000147889', 'ENSG00000147883',
             'ENSG00000123080','ENSG00000129355', 'ENSG00000149554','ENSG00000183765','ENSG00000005339','ENSG00000055130','ENSG00000006634',
             'ENSG00000101412','ENSG00000007968','ENSG00000112242','ENSG00000205250','ENSG00000133740','ENSG00000100393',
             'ENSG00000135476', 'ENSG00000105325', 'ENSG00000116717', 'ENSG00000099860', 'ENSG00000130222','ENSG00000082701','ENSG00000116478',
             'ENSG00000196591','ENSG00000002822','ENSG00000164109', 'ENSG00000116670','ENSG00000073111', 'ENSG00000112118',
             'ENSG00000104738', 'ENSG00000100297', 'ENSG00000076003','ENSG00000166508','ENSG00000135679',
            'ENSG00000136997', 'ENSG00000085840','ENSG00000115942','ENSG00000135336','ENSG00000115947', 'ENSG00000164815',
             'ENSG00000091651', 'ENSG00000132646', 'ENSG00000127564','ENSG00000166851','ENSG00000253729', 'ENSG00000164611',
             'ENSG00000250254','ENSG00000164754', 'ENSG00000139687','ENSG00000080839', 'ENSG00000103479', 'ENSG00000100387',
             'ENSG00000175793', 'ENSG00000113558','ENSG00000145604', 'ENSG00000175387','ENSG00000166949','ENSG00000141646',
             'ENSG00000072501','ENSG00000077935','ENSG00000108055','ENSG00000118007','ENSG00000101972','ENSG00000198176',
             'ENSG00000114126','ENSG00000105329','ENSG00000092969','ENSG00000119699','ENSG00000141510', 'ENSG00000112742',
             'ENSG00000166483', 'ENSG00000214102','ENSG00000166913','ENSG00000108953', 'ENSG00000170027','ENSG00000128245',
             'ENSG00000134308','ENSG00000164924','ENSG00000116809']
        elif organism =='mouse':
            cc_genes =     ['ENSMUSG00000026842', 'ENSMUSG00000014355', 'ENSMUSG00000036977','ENSMUSG00000025135', 'ENSMUSG00000035048', 'ENSMUSG00000026965',
               'ENSMUSG00000029176', 'ENSMUSG00000029472', 'ENSMUSG00000029466','ENSMUSG00000034218', 'ENSMUSG00000032409', 'ENSMUSG00000027379','ENSMUSG00000040084', 'ENSMUSG00000066979', 'ENSMUSG00000027793',
               'ENSMUSG00000027715', 'ENSMUSG00000041431', 'ENSMUSG00000032218','ENSMUSG00000051592', 'ENSMUSG00000116524', 'ENSMUSG00000070348',
               'ENSMUSG00000000184', 'ENSMUSG00000034165', 'ENSMUSG00000002068','ENSMUSG00000028212', 'ENSMUSG00000021548', 'ENSMUSG00000033502',
               'ENSMUSG00000033102', 'ENSMUSG00000038416', 'ENSMUSG00000006398','ENSMUSG00000024370', 'ENSMUSG00000032477', 'ENSMUSG00000027330',
               'ENSMUSG00000044201', 'ENSMUSG00000066149', 'ENSMUSG00000020687','ENSMUSG00000000028', 'ENSMUSG00000017499', 'ENSMUSG00000029283',
               'ENSMUSG00000019942', 'ENSMUSG00000025358', 'ENSMUSG00000006728','ENSMUSG00000040274', 'ENSMUSG00000069089', 'ENSMUSG00000023067',
               'ENSMUSG00000003031', 'ENSMUSG00000037664', 'ENSMUSG00000044303','ENSMUSG00000073802', 'ENSMUSG00000028551', 'ENSMUSG00000096472',
               'ENSMUSG00000032113', 'ENSMUSG00000029521', 'ENSMUSG00000022521','ENSMUSG00000029686', 'ENSMUSG00000002297', 'ENSMUSG00000027490',
               'ENSMUSG00000018983', 'ENSMUSG00000016477', 'ENSMUSG00000014859','ENSMUSG00000027552', 'ENSMUSG00000055024', 'ENSMUSG00000058290',
               'ENSMUSG00000020235', 'ENSMUSG00000036390', 'ENSMUSG00000015312','ENSMUSG00000021453', 'ENSMUSG00000022812', 'ENSMUSG00000028800',
               'ENSMUSG00000019777', 'ENSMUSG00000029554', 'ENSMUSG00000029910','ENSMUSG00000029003', 'ENSMUSG00000002870', 'ENSMUSG00000041859',
               'ENSMUSG00000022673', 'ENSMUSG00000005410', 'ENSMUSG00000026355','ENSMUSG00000029730', 'ENSMUSG00000020184', 'ENSMUSG00000022346',
               'ENSMUSG00000028587', 'ENSMUSG00000026037', 'ENSMUSG00000040044','ENSMUSG00000026761', 'ENSMUSG00000029012', 'ENSMUSG00000031697',
               'ENSMUSG00000027342', 'ENSMUSG00000023908', 'ENSMUSG00000030867','ENSMUSG00000022672', 'ENSMUSG00000020415', 'ENSMUSG00000022314',
               'ENSMUSG00000022105', 'ENSMUSG00000027641', 'ENSMUSG00000031666','ENSMUSG00000022400', 'ENSMUSG00000047281', 'ENSMUSG00000036309',
               'ENSMUSG00000054115', 'ENSMUSG00000111328', 'ENSMUSG00000024563','ENSMUSG00000032402', 'ENSMUSG00000024515', 'ENSMUSG00000041133',
               'ENSMUSG00000022432', 'ENSMUSG00000024974', 'ENSMUSG00000037286','ENSMUSG00000025862', 'ENSMUSG00000038482', 'ENSMUSG00000032411',
               'ENSMUSG00000002603', 'ENSMUSG00000039239', 'ENSMUSG00000021253','ENSMUSG00000059552', 'ENSMUSG00000038379', 'ENSMUSG00000031016',
               'ENSMUSG00000037159', 'ENSMUSG00000018326', 'ENSMUSG00000020849','ENSMUSG00000051391', 'ENSMUSG00000018965', 'ENSMUSG00000076432',
               'ENSMUSG00000022285', 'ENSMUSG00000006215']
        return cc_genes

    # encapsulate
    def get_mt_idx(self,genes):
            if not isinstance(genes,list): genes = list(genes)
            mt_genes = self.get_mito_genes(); idx  = []
            for i in mt_genes:
                if  i in genes: idx.append(genes.index(i))
            return idx
        
    #  encapsulate
    def get_cc_idx(self,genes):
            if not isinstance(genes,list): genes = list(genes)
            cc_genes = self.get_cell_cycle_genes(); idx=[]
            for i in cc_genes:
                if  i in genes: idx.append(genes.index(i))
            return idx
