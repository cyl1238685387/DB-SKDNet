# DB-SKDNet
DB-SKDNet: Efficient Semi-Supervised Change Detection via Dual-Branch Knowledge Alignment


## Requirements
<div style="background-color: #f6f8fa; border-radius: 3px; padding: 10px; margin: 5px 0;">
  <div style="text-align: right; margin-bottom: 5px;">
    <button onclick="copyToClipboard(this)"></button>
  </div>
  <pre style="margin: 0;">
- Pytorch 1.8.0  
- torchvision 0.9.0  
- python 3.8  
- opencv - python 4.5.3.56  
- tensorboardx 2.4  
- Cuda 11.3.1  
- Cudnn 11.3  
  </pre>
</div>

## Semi- Training or Test(WHU-CD)
<div style="background-color: #f6f8fa; border-radius: 3px; padding: 10px; margin: 5px 0;">
  <div style="text-align: right; margin-bottom: 5px;">
    <button onclick="copyToClipboard(this)"></button>
  </div>
  <pre style="margin: 0;">
python train.py --epoch 100 --batchsize 2 --gpu_id '1' --data_name 'WHU' --train_ratio 0.05 --model_name 'SemiModel_noema04'
python train.py --epoch 100 --batchsize 2 --gpu_id '1' --data_name 'WHU' --train_ratio 0.1 --model_name 'SemiModel_noema04'
python train.py --epoch 100 --batchsize 2 --gpu_id '1' --data_name 'WHU' --train_ratio 0.2 --model_name 'SemiModel_noema04'
python train.py --epoch 100 --batchsize 2 --gpu_id '1' --data_name 'WHU' --train_ratio 0.3 --model_name 'SemiModel_noema04'
<br><br>  
python test.py --gpu_id '1' --data_name 'WHU' --model_name 'SemiModel_noema04'
python test.py --gpu_id '1' --data_name 'WHU' --model_name 'SemiModel_noema04'
python test.py --gpu_id '1' --data_name 'WHU' --model_name 'SemiModel_noema04'
python test.py --gpu_id '1' --data_name 'WHU' --model_name 'SemiModel_noema04'

  </pre>
</div>


## Semi- Training or Test(LEVIR-CD)
<div style="background-color: #f6f8fa; border-radius: 3px; padding: 10px; margin: 5px 0;">
  <div style="text-align: right; margin-bottom: 5px;">
    <button onclick="copyToClipboard(this)"></button>
  </div>
  <pre style="margin: 0;">
python train.py --epoch 100 --batchsize 2 --gpu_id '1' --data_name 'LEVIR' --train_ratio 0.05 --model_name 'SemiModel_noema04'
python train.py --epoch 100 --batchsize 2 --gpu_id '1' --data_name 'LEVIR' --train_ratio 0.1 --model_name 'SemiModel_noema04'
python train.py --epoch 100 --batchsize 2 --gpu_id '1' --data_name 'LEVIR' --train_ratio 0.2 --model_name 'SemiModel_noema04'
python train.py --epoch 100 --batchsize 2 --gpu_id '1' --data_name 'LEVIR' --train_ratio 0.3 --model_name 'SemiModel_noema04'
<br><br>  
python test.py --gpu_id '1' --data_name 'LEVIR' --model_name 'SemiModel_noema04'
python test.py --gpu_id '1' --data_name 'LEVIR' --model_name 'SemiModel_noema04'
python test.py --gpu_id '1' --data_name 'LEVIR' --model_name 'SemiModel_noema04'
python test.py --gpu_id '1' --data_name 'LEVIR' --model_name 'SemiModel_noema04'
  </pre>
</div>


## Dataset Path Setting
<div style="background-color: #f6f8fa; border-radius: 3px; padding: 10px; margin: 5px 0;">
  <div style="text-align: right; margin-bottom: 5px;">
    <button onclick="copyToClipboard(this)"></button>
  </div>
  <pre style="margin: 0;">
 LEVIR-CD or WHU-CD  or GoogleGZ-CD
     |—train  
          |   |—A  
          |   |—B  
          |   |—label  
     |—val  
          |   |—A  
          |   |—B  
          |   |—label  
     |—test  
          |   |—A  
          |   |—B  
          |   |—label
  </pre>
</div>

<br><br> 
## Comparison with SOTA Methods
![image](https://github.com/user-attachments/assets/220de0aa-dc28-488b-b24f-33156a570b91)
Table 1. Quantitative Performance Comparison of State-of-the-Art Models at Differ-
ent Labeled Ratios on WHU-CD Dataset.
<br><br> 


![image](https://github.com/user-attachments/assets/aafeb1c3-206a-4bf7-9b6e-ac3d9e8435aa)
Table 2. Quantitative Performance Comparison of State-of-the-Art Models at Differ-
ent Labeled Ratios on LEVIR-CD Dataset.
<br><br> 

![image](https://github.com/user-attachments/assets/f06be317-90dc-426e-8b31-95378a0d97a8)
    Fig. 1. Comparison of different types of SSL methods.(1)Mean teacher (2)FixMatch.
<br><br>  

![image](https://github.com/user-attachments/assets/581eb07a-0395-4dee-8061-e6b5339cecac)
  Fig. 2. Comparison of different types of KD methods.
<br><br>  


![image](https://github.com/user-attachments/assets/df6987e5-8f28-4e2a-9367-1a143ed7dba6)
  Fig. 3. Overview of our proposed DB-SKDNet.
<br><br>  



![image](https://github.com/user-attachments/assets/a7cc6a14-fa8d-4d7e-8f90-79e0df54f613)
Fig. 4. the qualitative comparison results on WHU-CD dataset with 5%.
<br><br>  

![image](https://github.com/user-attachments/assets/49287ba0-f8f3-4623-b2b0-62d9baf7e426)
Fig. 5. the qualitative comparison results on WHU-CD dataset with 10%.
<br><br>  

![image](https://github.com/user-attachments/assets/6870cec3-dc38-4abd-9a91-4060a9158669)
Fig. 6. the qualitative comparison results on WHU-CD dataset with 20%.
<br><br>  

![image](https://github.com/user-attachments/assets/27e36bb0-80ff-4f0f-8837-8a37ce24910d)
Fig. 7. the qualitative comparison results on WHU-CD dataset with 30%.
<br><br>  

![image](https://github.com/user-attachments/assets/8bfdc04f-9e2b-4fca-a01d-b78590151969)
Fig. 8. the qualitative comparison results on LEVIR-CD dataset with 5%.
<br><br>  

![image](https://github.com/user-attachments/assets/d844ba78-a37e-4010-95fa-7f02a34fb3b0)
Fig. 9. the qualitative comparison results on LEVIR-CD dataset with 10%.
<br><br>  

![image](https://github.com/user-attachments/assets/e70b45ad-b3ea-4f50-985a-a3f03d7106eb)
Fig. 10. the qualitative comparison results on LEVIR-CD dataset with 20%.
<br><br>  

![image](https://github.com/user-attachments/assets/e8bea60e-b03a-4b24-8e34-771d12e64e02)
Fig. 11. the qualitative comparison results on LEVIR-CD dataset with 30%.
<br><br>  
