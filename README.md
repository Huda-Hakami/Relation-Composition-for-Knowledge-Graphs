# Relation-Composition-for-Knowledge-Graphs (RelComp)
Learning compositional functions for representing relations in knowledge graphs.

This repository contains the code and data for the paper: Wenye Chen, Huda Hakami and Danushka Bollegala: [Learning to Compose Relational Embeddings in Knowledge Graphs](https://cgi.csc.liv.ac.uk/~huda/papers/relinf.pdf) Proc. of the 16th International Conference of the Pacific Association for Computational Linguistics (PACLING), October, 2019.

# Prerequisites
To implementation code in this project requires:

     - Python
     - sklearnt 
     - tensorflow
     - matplotlib
     
# Data
This project contains the following data files:]

- FB14K-473 folder: includes FB15K-474 data, which is an extension of FB15K-237 by inclusing reverse triples in train/test/valid splits. The folder includes entity2id, relations2id, train, test, valid splits.  
- RelWalk_Embeddings: includes pretrained RelWalk entity and relation embeddings for FB15K-474.A python script to read such embeddings is in the folder (Read_Embeddings.py)
- Composition-constraints: includes relational compositional constraints where two relations r_A and r_B jointly implies a third relation r_C. This constraints are generated by Takahashi et al. (2018) in their paper: [Interpretable and Compositional Relation Learning by Joint Training with an Autoencoder](https://www.aclweb.org/anthology/P18-1200/). Each line in relcompstats_filter.txt file corresponds to one constraint that written the format: ***r_A r_B r_C Jacard_score cardinality_of_intersection*** . relcomp2id.txt mapps each constraint in relcompstats_filter.txt using relation-ids in FB15K-474 in the format: ***(r_A_id, r_B_id, r_C_id)*** . 

# Cite
If you use this code, please cite this paper as follows.

     @inproceedings{Chen:PACLING:2019,    
         title={Learning to Compose Relational Embeddings in Knowledge Graphs},    
         author={Wenye Chen and Huda Hakami and Danushka Bollegala},    
         booktitle={Proc. of the 16th International Conference of the Pacific Association for Computational Linguistics (PACLING)},    
         year={2019} 
       }

