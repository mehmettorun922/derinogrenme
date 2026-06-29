# -*- coding: utf-8 -*-
https://colab.research.google.com/drive/1G0zyMs4xtJp8WKKJmOrdabOEuCsjv3w8


import yaml
from pathlib import Path


extract_dir = CFG['extract_dir']
data_yaml_path = Path(extract_dir) / 'data.yaml'

translations = {
    'Apple Scab Leaf': 'Elma Kabuk Hastalığı Yaprağı',
    'Apple leaf': 'Elma Yaprağı',
    'Apple rust leaf': 'Elma Pas Hastalığı Yaprağı',
    'Bell_pepper leaf spot': 'Biber Yaprak Lekesi',
    'Bell_pepper leaf': 'Biber Yaprağı',
    'Blueberry leaf': 'Yaban Mersini Yaprağı',
    'Cherry leaf': 'Kiraz Yaprağı',
    'Corn Gray leaf spot': 'Mısır Gri Yaprak Lekesi',
    'Corn leaf blight': 'Mısır Yaprak Yanıklığı',
    'Corn rust leaf': 'Mısır Pas Hastalığı Yaprağı',
    'Peach leaf': 'Şeftali Yaprağı',
    'Potato leaf early blight': 'Patates Erken Yanıklık Yaprağı',
    'Potato leaf late blight': 'Patates Geç Yanıklık Yaprağı',
    'Potato leaf': 'Patates Yaprağı',
    'Raspberry leaf': 'Ahududu Yaprağı',
    'Soyabean leaf': 'Soya Fasulyesi Yaprağı',
    'Soybean leaf': 'Soya Fasulyesi Yaprağı',
    'Squash Powdery mildew leaf': 'Kabak Külleme Yaprağı',
    'Strawberry leaf': 'Çilek Yaprağı',
    'Tomato Early blight leaf': 'Domates Erken Yanıklık Yaprağı',
    'Tomato Septoria leaf spot': 'Domates Septoria Yaprak Lekesi',
    'Tomato leaf bacterial spot': 'Domates Yaprak Bakteriyel Lekesi',
    'Tomato leaf late blight': 'Domates Geç Yanıklık Yaprağı',
    'Tomato leaf mosaic virus': 'Domates Yaprak Mozaik Virüsü',
    'Tomato leaf yellow virus': 'Domates Yaprak Sarı Virüsü',
    'Tomato leaf': 'Domates Yaprağı',
    'Tomato mold leaf': 'Domates Küf Yaprağı',
    'Tomato two spotted spider mites leaf': 'Domates İki Noktalı Kırmızı Örümcek Yaprağı',
    'grape leaf black rot': 'Üzüm Yaprağı Kara Çürüklüğü',
    'grape leaf': 'Üzüm Yaprağı'
}

if data_yaml_path.exists():
    with open(data_yaml_path, 'r') as f:
        data_yaml = yaml.safe_load(f)

    if 'names' in data_yaml and isinstance(data_yaml['names'], list):
        updated_names = []
        for name in data_yaml['names']:
            turkish_name = translations.get(name, name) 
            updated_names.append(f'{name} ({turkish_name})')
        data_yaml['names'] = updated_names

        with open(data_yaml_path, 'w') as f:
            yaml.dump(data_yaml, f, allow_unicode=True, sort_keys=False)
        print(f' {data_yaml_path} dosyası Türkçe çevirilerle güncellendi.')
    else:
        print(f' {data_yaml_path} içinde "names" anahtarı veya liste formatı bulunamadı. Güncelleme yapılamadı.')
else:
    print(f' {data_yaml_path} bulunamadı. Lütfen dosya yolunu kontrol edin.')

print("\nDeğişikliklerin etkili olması için lütfen ana kod hücresini (7cM2DUM88Lar) yeniden çalıştırın.")

import torch
ckpt = torch.load(CFG['save_path'])
ckpt['classes'] = classes
torch.save(ckpt, CFG['save_path'])
print(' Sınıf listesi (Türkçe çevirilerle birlikte) best_model.pth dosyasına kaydedildi.')
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, WeightedRandomSampler, Subset, Dataset
from torchvision import models
import numpy as np
from pathlib import Path
import zipfile, os, time
from collections import Counter
from PIL import Image
import yaml 

try:
    from sklearn.metrics import classification_report, confusion_matrix
    import matplotlib.pyplot as plt
    import seaborn as sns
    import yaml
except ImportError:
    os.system("pip install scikit-learn seaborn pyyaml -q") 
    from sklearn.metrics import classification_report, confusion_matrix
    import matplotlib.pyplot as plt
    import seaborn as sns
    import yaml

import warnings; warnings.filterwarnings('ignore')


device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
USE_AMP = device.type == 'cuda'   
SEED    = 42
torch.manual_seed(SEED); np.random.seed(SEED)
if USE_AMP: torch.cuda.manual_seed_all(SEED)

print(f'{"="*60}')
print(f'  Cihaz  : {device}')
if USE_AMP:
    print(f'  GPU    : {torch.cuda.get_device_name(0)}')
    print(f'  Bellek : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB')
else:
    print('    GPU yok! → Colab: Çalışma Zamanı > Donanım Hızlandırıcısı > T4 GPU')
print(f'{"="*60}\n')


CFG = dict(
    zip_path     = 'plantdoc.zip',    
    extract_dir  = './plantdoc_data',
    img_size     = 224,
    batch_size   = 32,         
    epochs       = 40,
    lr           = 3e-4,       
    backbone_lr  = 3e-5,       
    weight_decay = 5e-4,
    patience     = 8,          
    warmup       = 5,        
    val_split    = 0.20,        

    model        = 'efficientnet_b3',
    save_path    = 'best_model.pth',
    workers      = 2          
)


if not os.path.exists(CFG['extract_dir']):
    print(f' {CFG["zip_path"]} çıkarılıyor...')
    try:
        with zipfile.ZipFile(CFG['zip_path'], 'r') as z:
            z.extractall(CFG['extract_dir'])
        print('Çıkarma tamamlandı.\n')
    except FileNotFoundError:
        print(f' "{CFG["zip_path"]}" bulunamadı! '
              'Colab sol menüsünden dosyalar sekmesine tam bu adla yükle.')
        raise
else:
    print(' Veri zaten çıkarılmış.\n')


root = Path(CFG['extract_dir'])

def _find_data_root(base: Path) -> Path:

    train_candidate = base / 'train'
    if train_candidate.is_dir():

        if (train_candidate / 'images').is_dir() and (train_candidate / 'labels').is_dir():
            return train_candidate.parent 


    img_dirs = [d for d in base.rglob('images') if d.is_dir()]
    if img_dirs:

        return img_dirs[0].parent.parent if img_dirs[0].parent.name == 'train' else img_dirs[0].parent
    return base 

main_data_root = _find_data_root(root)

train_dir_imgs = main_data_root / 'train' / 'images'
train_dir_labels = main_data_root / 'train' / 'labels'

val_dir_imgs = main_data_root / 'valid' / 'images' 
val_dir_labels = main_data_root / 'valid' / 'labels'

test_dir_imgs = main_data_root / 'test' / 'images'
test_dir_labels = main_data_root / 'test' / 'labels'


all_class_ids = set()
for label_file in train_dir_labels.glob('*.txt'):
    with open(label_file, 'r') as f:
       
        for line in f:
            class_id = line.strip().split(' ')[0]
            all_class_ids.add(class_id)


class_id_strings = sorted(list(all_class_ids))
num_classes = len(class_id_strings)

if not class_id_strings:
    raise RuntimeError("Etiket dosyalarında hiçbir sınıf ID'si bulunamadı. Lütfen etiket dosyalarının formatını kontrol edin.")


id_to_descriptive_name = {}
data_yaml_path = Path(CFG['extract_dir']) / 'data.yaml'

if data_yaml_path.exists():
    try:
        with open(data_yaml_path, 'r') as f:
            data_yaml = yaml.safe_load(f)
            if 'names' in data_yaml and isinstance(data_yaml['names'], list):
                for i, name in enumerate(data_yaml['names']):
                    id_to_descriptive_name[str(i)] = name 
                print(f' Sınıf isimleri {data_yaml_path} dosyasından yüklendi.\n')
            else:
                print(f' {data_yaml_path} içinde "names" anahtarı veya liste formatı bulunamadı. Sayısal ID\'ler kullanılacak.\n')
    except Exception as e:
        print(f' {data_yaml_path} okunurken hata oluştu: {e}. Sayısal ID\'ler kullanılacak.\n')
else:
    print(f' {data_yaml_path} bulunamadı, sayısal ID\'ler kullanılacak.\n')


class_to_idx = {class_id: i for i, class_id in enumerate(class_id_strings)}


classes = [id_to_descriptive_name.get(class_id, class_id) for class_id in class_id_strings]
idx_to_class = {i: descriptive_name for i, descriptive_name in enumerate(classes)}

print(f' Train images dizini  : {train_dir_imgs}')
print(f' Train labels dizini  : {train_dir_labels}')
print(f' Test images dizini   : {test_dir_imgs if test_dir_imgs.is_dir() else "(yok)"}')
print(f' Test labels dizini   : {test_dir_labels if test_dir_labels.is_dir() else "(yok)"}')
print(f' Sınıf sayısı         : {num_classes}')
print(f' İlk 4 sınıf          : {classes[:4]}...\n')

class PlantDocDataset(Dataset):
    def __init__(self, img_dir, label_dir, class_to_idx, transform=None):
        self.img_dir = Path(img_dir)
        self.label_dir = Path(label_dir)
        self.class_to_idx = class_to_idx 
        self.transform = transform

        self.samples = []
        for img_path in sorted(list(self.img_dir.glob('*.jpg')) + list(self.img_dir.glob('*.jpeg')) + list(self.img_dir.glob('*.png'))):
            label_filename = img_path.stem + '.txt'
            label_path = self.label_dir / label_filename

            if label_path.exists():
                with open(label_path, 'r') as f:
                    labels_in_file = f.readlines()

                    if labels_in_file:
                        class_id_str = labels_in_file[0].strip().split(' ')[0] 
                        if class_id_str in self.class_to_idx:
                            class_idx = self.class_to_idx[class_id_str] 
                            self.samples.append((img_path, class_idx))
                        else:
                            print(f"Uyarı: {label_path} dosyasında bilinmeyen sınıf ID'si: {class_id_str}")
                    else:
                        print(f"Uyarı: {label_path} boş etiket dosyası.")
            else:
                print(f"Uyarı: {label_path} için etiket dosyası bulunamadı, görsel atlandı: {img_path}")

        if not self.samples:
            raise FileNotFoundError(f"Belirtilen dizinlerde (img: {img_dir}, label: {label_dir}) eşleşen resim ve etiket çifti bulunamadı.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label


SZ   = CFG['img_size']
MEAN = [0.485, 0.456, 0.406]   
STD  = [0.229, 0.224, 0.225]

train_tf = transforms.Compose([
    transforms.Resize((SZ + 32, SZ + 32)),     
    transforms.RandomCrop(SZ),
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomVerticalFlip(0.2),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3,
                           saturation=0.3, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
    transforms.RandomErasing(p=0.15, scale=(0.02, 0.25)), 
])

val_tf = transforms.Compose([
    transforms.Resize((SZ, SZ)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


full_train_ds = PlantDocDataset(img_dir=train_dir_imgs, label_dir=train_dir_labels, class_to_idx=class_to_idx, transform=train_tf)
full_val_ds   = PlantDocDataset(img_dir=val_dir_imgs, label_dir=val_dir_labels, class_to_idx=class_to_idx, transform=val_tf) if val_dir_imgs.is_dir() else None
full_test_ds  = PlantDocDataset(img_dir=test_dir_imgs, label_dir=test_dir_labels, class_to_idx=class_to_idx, transform=val_tf) if test_dir_imgs.is_dir() else None

if full_val_ds is None:
    print("\n⚠️ Ayrı validasyon seti bulunamadı, eğitim setinden bölünüyor.")
    n_train = len(full_train_ds)
    n_tr    = int((1 - CFG['val_split']) * n_train)
    n_val   = n_train - n_tr

    perm   = torch.randperm(n_train, generator=torch.Generator().manual_seed(SEED)).tolist()
    tr_idx = perm[:n_tr]
    va_idx = perm[n_tr:]

    train_ds = PlantDocDataset(img_dir=train_dir_imgs, label_dir=train_dir_labels, class_to_idx=class_to_idx, transform=train_tf)
    val_ds   = PlantDocDataset(img_dir=train_dir_imgs, label_dir=train_dir_labels, class_to_idx=class_to_idx, transform=val_tf)

    train_ds = Subset(train_ds, tr_idx)
    val_ds   = Subset(val_ds, va_idx)
    test_ds  = val_ds

else:
    train_ds = full_train_ds
    val_ds   = full_val_ds
    test_ds  = full_test_ds if full_test_ds else full_val_ds 



if isinstance(train_ds, Subset):
    
    original_targets = [s[1] for s in train_ds.dataset.samples] if hasattr(train_ds.dataset, 'samples') else []
    tr_labels = [original_targets[i] for i in tr_idx]
else:
    tr_labels = [s[1] for s in train_ds.samples]


cls_cnt   = Counter(tr_labels)
smp_w     = [1.0 / cls_cnt[l] for l in tr_labels]
sampler   = WeightedRandomSampler(
    weights=torch.tensor(smp_w, dtype=torch.float),
    num_samples=len(smp_w),
    replacement=True
)

pw = CFG['workers'] > 0   
trainloader = DataLoader(train_ds, batch_size=CFG['batch_size'], sampler=sampler,
                         num_workers=CFG['workers'], pin_memory=True, persistent_workers=pw)
valloader   = DataLoader(val_ds,   batch_size=CFG['batch_size'], shuffle=False,
                         num_workers=CFG['workers'], pin_memory=True, persistent_workers=pw)
testloader  = DataLoader(test_ds,  batch_size=CFG['batch_size'], shuffle=False,
                         num_workers=CFG['workers'], pin_memory=True, persistent_workers=pw)

print(f'\n✓ Train: {len(train_ds):5d} | Val: {len(val_ds):5d} | Test: {len(test_ds):5d}')
print(f'✓ Batch: {len(trainloader)} train / {len(valloader)} val\n')


def build_model(name: str, nc: int) -> nn.Module:
    """
    ImageNet ile pretrained modeli yükle, son katmanı değiştir.
    Backbone'u dondurmuyoruz → fine-tuning (backbone_lr daha düşük tutulur).
    """
    if name == 'efficientnet_b3':
        m = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1)
        in_f = m.classifier[1].in_features  
        m.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_f, 512),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(512, nc),
        )
    elif name == 'resnet50':
        m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        in_f = m.fc.in_features 
        m.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_f, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, nc),
        )
    else:
        raise ValueError(f"'{name}' geçersiz. 'efficientnet_b3' veya 'resnet50' girin.")
    return m

model = build_model(CFG['model'], num_classes).to(device)
print(f'✓ Model         : {CFG["model"]}')
print(f'  Toplam param  : {sum(p.numel() for p in model.parameters()):,}')
print(f'  Eğitilebilir  : {sum(p.numel() for p in model.parameters() if p.requires_grad):,}\n')


w = torch.tensor(
    [1.0 / cls_cnt.get(i, 1) for i in range(num_classes)], dtype=torch.float
).to(device)
w = w / w.sum() * num_classes  
criterion = nn.CrossEntropyLoss(weight=w, label_smoothing=0.05)

backbone_p, head_p = [], []
for pname, p in model.named_parameters():
    if any(k in pname for k in ('classifier', 'fc')):
        head_p.append(p)
    else:
        backbone_p.append(p)

optimizer = optim.AdamW(
    [{'params': backbone_p, 'lr': CFG['backbone_lr']},
     {'params': head_p,     'lr': CFG['lr']}],
    weight_decay=CFG['weight_decay']
)


from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
WU         = CFG['warmup']
sch_warm   = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=WU)
sch_cos    = CosineAnnealingLR(optimizer, T_max=max(CFG['epochs'] - WU, 1), eta_min=1e-7)
scheduler  = SequentialLR(optimizer, [sch_warm, sch_cos], milestones=[WU])

scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)


def run_epoch(model, loader, optimizer, criterion, scaler, train: bool):
    """
    Tek epoch eğitim (train=True) veya değerlendirme (train=False) yapar.
    Loss, accuracy, tahminler ve gerçek etiketleri döndürür.
    """
    model.train(train)
    total_loss = correct = total = 0
    all_pred, all_true = [], []

    with torch.set_grad_enabled(train):
        for imgs, lbls in loader:
            imgs = imgs.to(device, non_blocking=True)
            lbls = lbls.to(device, non_blocking=True)

            if train:
                optimizer.zero_grad(set_to_none=True)

        
            with torch.cuda.amp.autocast(enabled=USE_AMP):
                out  = model(imgs)
                loss = criterion(out, lbls)

            if train:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  
                scaler.step(optimizer)
                scaler.update()

            pred = out.argmax(1)
            total_loss += loss.item() * imgs.size(0)
            correct    += (pred == lbls).sum().item()
            total      += imgs.size(0)
            all_pred.extend(pred.cpu().numpy())
            all_true.extend(lbls.cpu().numpy())

    return total_loss / total, correct / total, all_pred, all_true


hist = {k: [] for k in ('tl', 'ta', 'vl', 'va', 'lr')}
best_acc = best_ep = patience_cnt = 0

print(f'{"═"*70}')
print(f'  EĞİTİM  |  max {CFG["epochs"]} Epoch  |  {CFG["model"]}  |  {device}')
print(f'{"═"*70}')
print(f'{"Ep":>4}  {"TrLoss":>7}  {"TrAcc":>6}  {"VaLoss":>7}  {"VaAcc":>6}  {"LR":>9}  {"s":>5}')
print('─' * 70)

for ep in range(1, CFG['epochs'] + 1):
    t0 = time.time()

    tl, ta, _, _           = run_epoch(model, trainloader, optimizer, criterion, scaler, train=True)
    vl, va, val_pred, val_true = run_epoch(model, valloader, optimizer, criterion, scaler, train=False)
    scheduler.step()

    lr = optimizer.param_groups[1]['lr']   
    dt = time.time() - t0

    for k, v in zip(('tl','ta','vl','va','lr'), (tl, ta, vl, va, lr)):
        hist[k].append(v)

    flag = '  ' if va > best_acc else ''
    print(f'{ep:4d}  {tl:7.4f}  {ta*100:5.1f}%  {vl:7.4f}  {va*100:5.1f}%  {lr:9.2e}  {dt:5.1f}s{flag}')

    if va > best_acc:
        best_acc, best_ep, patience_cnt = va, ep, 0
        torch.save({
            'epoch'      : ep,
            'val_acc'    : va,
            'model_state': model.state_dict(),
            'optim_state': optimizer.state_dict(),
            'classes'    : classes, 
            'config'     : CFG
        }, CFG['save_path'])
    else:
        patience_cnt += 1
        if patience_cnt >= CFG['patience']:
            print(f'\n⏹  Early Stop — {CFG["patience"]} epoch boyunca iyileşme olmadı.')
            break

print(f'\n Eğitim tamamlandı  |  Best Val: %{best_acc*100:.2f}  (Epoch {best_ep})')
print(f'   Kaydedildi → {CFG["save_path"]}\n')


ckpt = torch.load(CFG['save_path'], map_location=device)
model.load_state_dict(ckpt['model_state'])

print(f'📦 En iyi model yüklendi (Epoch {ckpt["epoch"]}, Val %{ckpt["val_acc"]*100:.2f})\n')

_, test_acc, tpred, ttrue = run_epoch(model, testloader, optimizer, criterion, scaler, train=False)
print(f'🎯 Test Accuracy : %{test_acc*100:.2f}\n')
print(classification_report(ttrue, tpred, target_names=classes, digits=3, zero_division=0)) 


eps_x = range(1, len(hist['tl']) + 1)
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle(f'PlantDoc | {CFG["model"]} | Best Val %{best_acc*100:.2f}',
             fontsize=14, fontweight='bold')

ax = axes[0]
ax.plot(eps_x, hist['tl'], label='Train', color='#E74C3C', lw=2)
ax.plot(eps_x, hist['vl'], label='Val',   color='#3498DB', lw=2)
ax.set(title='Loss Eğrisi', xlabel='Epoch', ylabel='Loss')
ax.legend(); ax.grid(alpha=0.3)


ax = axes[1]
ax.plot(eps_x, [a*100 for a in hist['ta']], label='Train', color='#E74C3C', lw=2)
ax.plot(eps_x, [a*100 for a in hist['va']], label='Val',   color='#3498DB', lw=2)
ax.axhline(best_acc*100, color='#27AE60', ls='--', alpha=0.8,
           label=f'Best {best_acc*100:.1f}%')
ax.set(title='Doğruluk Eğrisi', xlabel='Epoch', ylabel='Accuracy (%)')
ax.legend(); ax.grid(alpha=0.3)

ax = axes[2]
ax.plot(eps_x, hist['lr'], color='#9B59B6', lw=2)
ax.set(title='Learning Rate Eğrisi', xlabel='Epoch', ylabel='LR')
ax.set_yscale('log'); ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('training_curves.png', dpi=150, bbox_inches='tight')
plt.show()


fig, ax = plt.subplots(figsize=(max(10, num_classes // 2), max(9, num_classes // 2)))
cm = confusion_matrix(ttrue, tpred)
sns.heatmap(cm, annot=(num_classes <= 30), fmt='d', cmap='Blues',
            xticklabels=classes, yticklabels=classes, ax=ax, annot_kws={'size': 7}) 
ax.set_title('Confusion Matrix', fontsize=13, fontweight='bold')
ax.set_ylabel('Gerçek Sınıf'); ax.set_xlabel('Tahmin Edilen')
plt.xticks(rotation=45, ha='right', fontsize=7)
plt.yticks(fontsize=7)
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.show()

print('📊 Grafikler kaydedildi: training_curves.png | confusion_matrix.png')



def predict(img_path: str, top_k: int = 5):
    """
    Tek bir görseli sınıflandır ve top-k sonucu yazdır.

    Kullanım:
        predict('yaprak.jpg')           # top-5
        predict('yaprak.jpg', top_k=3) # top-3
    """

    ckpt = torch.load(CFG['save_path'], map_location=device)
    model.load_state_dict(ckpt['model_state']) 


    current_classes = ckpt['classes'] 

    tf = transforms.Compose([
        transforms.Resize((SZ, SZ)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    img  = Image.open(img_path).convert('RGB')
    inp  = tf(img).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        probs = torch.softmax(model(inp), dim=1)

    top_p, top_i = probs.topk(min(top_k, len(current_classes))) 
    print(f'\n🌿  Tahmin: {img_path}')
    print('─' * 52)
    for p, i in zip(top_p[0], top_i[0]):
        bar = '█' * int(p.item() * 30)

        print(f'  {current_classes[i.item()]:38s} {bar} {p.item()*100:5.1f}%')

    best_class = current_classes[top_i[0][0].item()]
    best_prob  = top_p[0][0].item()
    print(f'\n  → Sonuç: {best_class}  (%{best_prob*100:.1f})')
    return best_class, best_prob


print('\n\n' + '═'*55)
print('  HER ŞEY TAMAMLANDI!')
print(f'     En İyi Val Acc  : %{best_acc*100:.2f}  (Epoch {best_ep})')
print(f'     Model           : {CFG["save_path"]}')
print(f'     Tahmin yapmak   : predict("gorsel.jpg")')
print('═'*55)

predict('yaprak.jpg')


print(f'🔍 Test seti değerlendiriliyor: {len(testloader.dataset)} görsel...')


ckpt = torch.load(CFG['save_path'], map_location=device)
model.load_state_dict(ckpt['model_state'])

_, test_acc, tpred, ttrue = run_epoch(model, testloader, optimizer, criterion, scaler, train=False)

print(f'\n Toplam Test Başarımı (Accuracy): %{test_acc*100:.2f}')
print('\n' + '='*30)
print('DETAYLI SINIFLANDIRMA RAPORU')
print('='*30)
print(classification_report(ttrue, tpred, target_names=classes, digits=3, zero_division=0))

import random
from pathlib import Path

selected_test_img = '/content/plantdoc_data/test/images/0000_jpg.rf.7bf8d4c69ad253ee55c87d6e78d1ae28.jpg'


train_images_path = Path('/content/plantdoc_data/train/images')
all_train_images = list(train_images_path.glob('*.jpg')) + list(train_images_path.glob('*.png'))
random_train_samples = random.sample(all_train_images, 3)

print("--- Seçilen Test Görseli Tahmini ---")
predict(selected_test_img)

print("\n--- Eğitim Setinden Rastgele Örnek Tahminleri ---")
for img_p in random_train_samples:
    predict(str(img_p))

predict('/content/plantdoc_data/train/images/image_0000.jpg')

predict('/content/plantdoc_data/train/images/image_0011.jpg')
