# GunPose
## Overview
Proyek ini bertujuan untuk mengembangkan sistem deteksi pose yang mampu mengidentifikasi secara real-time situasi berbahaya seperti orang yang menembak menggunakan senjata api, bertengkar, bertarung menggunakan pedang, atau menggunakan golok. Sistem ini dirancang untuk mendukung upaya peningkatan keamanan di berbagai lingkungan, seperti area publik, tempat kerja, atau zona rawan konflik.

## Dataset
Dataset didapatkan dari berbagai sumber pada internet yang memiliki gambar posisi penembak, pertengkaran, pertarungan pedang dan golok yang bervariasi. Persebaran data yang digunakan adalah sebagai berikut:  
| Pose  | Train | Test | 
| ------------- | ------------- | ------------- |
| Gun  | 361 | 90 |
| Fight  | 616 | 154 |
| Sword  | 409 | 102 |
| Golok  | 475 | 118 |


## Model 
Proyek ini menggunakan model XGBoost dengan mempelajari titik-titik _skeleton_ yang didapatkan dari model YOLO11-Pose.

### Environment
- GeForce RTX 4060
- Python 3.12.1
- Pytorch 2.5.1
- Torchvision 0.20.1
- Torchaudio 2.5.1
- Ultralytics 8.3.39
- xgboost 2.1.1
- scikit-learn 1.3.2
- pandas 2.1.4
- opencv 4.10.0.84

### Metrik Evaluasi
| Model | Akurasi  | 
| ------------- | ------------- | 
| Gun | 92.03 |
| Fight | 94.76 |
| Sword | 91.86 |
| Golok | 96.87 |

### Hasil
#### Gun
![image](https://github.com/user-attachments/assets/47d3a997-da89-4666-8648-3242d3c9435d)

#### Fight
![image](https://github.com/user-attachments/assets/aa00f42e-97bf-43de-8f6a-339fb4edf977)

#### Sword
![image](https://github.com/user-attachments/assets/d03dc813-54e5-4eaa-b1df-5df39becef63)

#### Golok
![image](https://github.com/user-attachments/assets/ad31de4f-d425-4217-a551-f926052f026f)
