Hierarchical Networks With Subnetwork Nodes
----------------------------


Description: 

 Iterative method of learning has become a paradigm for training hierarchical NNs. they have improved accuracies of objection recognition and have turned out to be very good at
discovering intricate structures. however it is clear that the learning effectiveness and learning speed of these networks are far slower than required, which has been a
major bottleneck for many applications. iterative methods such as back propagation used in DL suffer issues like slow convergence. HSNN a deep three-general layer learning framework is used to overcome these shortcomings. 

Development:

 The Structure of HSNN has 3 main stages:
   - Subspace Feature Extraction: 2 layer subnetworks used for Feature extraction
   - Feature Combination: combining Extracted Features
   - Classification: ELM can be used for final classification

Dataset:
 
 Classification datasets with all inputs having same number of features should be used. training and testing data should be given as 2 CSV files. 
   - Expected Output: 1st colomn contains output are arranged in column before Output
   - Data: Remaining coloumns should have Features.
Example : Scene-15 SpatialDataset (https://drive.google.com/drive/folders/1HCv_Wh4evDpsMeXs0HSNXmdBRm4aoGxR?usp=sharing)


Main module & function:

 - LoadData() is the wrapper function that load training and testing data and one hot encodes the labels.
 - subnetwrork(): is the wrapper function used for Feature Extaction
 - featurecomb(): is the wrapper function used for combining Features
 - lastlayer(): is the wrapper function used for final classification.

Example usage:
 
  >>> T, P, TV_T, TV_P, repeat = loadata('Scene_15_train.csv','Scene_15_test.csv')
  >>> NumberofHiddenNeurons=80
  >>> C=128
  >>> kkkk=2
  >>> sn=3
  >>> name='scene15_channel_'
  >>> chnl=1
  >>> load
  >>> for i in range(chnl):
  >>>   start_time = time.time()   
  >>>   subnetwrork(P,T,TV_P,TV_T, NumberofHiddenNeurons, C, kkkk, sn, name, i+1)
  >>>   print("Subspace Feature Extraction Time for channel_"+str(i+1)+" : ",time.time()-start_time,"\n")
  >>>   start_time = time.time() 
  >>>   Features = featurecomb(name, sn, chnl)
  >>>   print("Combining Feature Time Taken : ",time.time()-start_time,"\n")
  >>> Target = np.genfromtxt(name+str(chnl)+"target.csv", delimiter=',')
  >>> C2=4096
  >>> No_train = len(P[1])
  >>> start_time = time.time()
  >>> lastlayer(Features,Target,C2,No_train)
  >>> print("Final Classification Time Taken : ",time.time()-start_time)
  
Authors & Acknowledgements:
 
 - See the included AUTHORS file for more information.
 - Special thanks to Yimin Yang, Comp. Sc, Lakehead University 
  
License:
 
 This software is licensed under the BSD License. See the included LICENSE file for more information.


References:
 1. Y. Yang and Q. M. J. Wu, "Features Combined From Hundreds of Midlayers: Hierarchical Networks With Subnetwork Nodes," in IEEE Transactions on Neural Networks and Learning Systems, vol. 30, no. 11, pp. 3313-3325, Nov. 2019, doi: 10.1109/TNNLS.2018.2890787.
 2. Huang, Guang-Bin & Zhu, Qin-Yu & Siew, Chee. (2004). Extreme learning machine: A new learning scheme of feedforward neural networks. IEEE International Conference on Neural Networks - Conference Proceedings. 2. 985 - 990 vol.2. 10.1109/IJCNN.2004.1380068.
 3. http://www.yiminyang.com/publications.html
 4. https://docs.nvidia.com/cuda/
 5. https://paperswithcode.com/method/max-pooling
