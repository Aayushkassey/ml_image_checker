import os
import shutil

base_dir = 'ml_waste_system'
classes = ['high_waste', 'invalid_or_clean', 'low_waste', 'medium_waste']
valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

print("🔄 `val` फोल्डरका सबै फोटोहरूलाई `train` मा फिर्ता ल्याइँदैछ...\n")

for cls in classes:
    val_cls_dir = os.path.join(base_dir, 'val', cls)
    train_cls_dir = os.path.join(base_dir, 'train', cls)

    os.makedirs(train_cls_dir, exist_ok=True)

    if os.path.exists(val_cls_dir):
        val_images = [
            f for f in os.listdir(val_cls_dir)
            if os.path.isfile(os.path.join(val_cls_dir, f)) and os.path.splitext(f)[1].lower() in valid_extensions
        ]
        
        for img in val_images:
            src_path = os.path.join(val_cls_dir, img)
            dst_path = os.path.join(train_cls_dir, img)
            shutil.move(src_path, dst_path)
            
        print(f"✅ {cls}: {len(val_images)} फोटोहरू val बाट train मा फिर्ता सारियो।")

print("\n🎉 Reset सकियो! अब सबै फोटोहरू `train` फोल्डरमा जम्मा भए।")