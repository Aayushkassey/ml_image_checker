import os
import shutil
import random

# Source र Destination paths (तपाईँकै फोल्डर स्ट्रक्चर अनुसार)
base_dir = 'ml_waste_system'
classes = ['low_waste', 'medium_waste', 'high_waste']
split_ratio = 0.8  # 80% Train, 20% Val

print("🔄 `ml_waste_system/train` बाट फोटोहरू ८०/२० मा मिलाउने काम सुरु भयो...\n")

for cls in classes:
    train_cls_dir = os.path.join(base_dir, 'train', cls)
    val_cls_dir = os.path.join(base_dir, 'val', cls)

    # Val फोल्डर छैन भने बनाउने
    os.makedirs(val_cls_dir, exist_ok=True)

    if not os.path.exists(train_cls_dir):
        print(f"⚠️ Warning: {train_cls_dir} भेटिएन!")
        continue

    # Train फोल्डर भित्र भएका सबै फोटोहरूको लिस्ट लिने
    images = [f for f in os.listdir(train_cls_dir) if os.path.isfile(os.path.join(train_cls_dir, f))]

    if len(images) == 0:
        print(f"⚠️ {cls} भित्र कुनै पनि फोटो भेटिएन!")
        continue

    # फोटोहरू र्‍यान्डम्ली घालमेल गर्ने
    random.shuffle(images)

    # ८०/२० को हिसाब निकाल्ने
    train_count = int(len(images) * split_ratio)
    val_images = images[train_count:]  # बाँकी २०% फोटोहरू Val मा सार्नका लागि

    # २०% फोटोहरूलाई Train बाट सारेर Val मा हाल्ने (Move गर्ने)
    for img in val_images:
        src_path = os.path.join(train_cls_dir, img)
        dst_path = os.path.join(val_cls_dir, img)
        shutil.move(src_path, dst_path)

    remaining_train_count = len(images) - len(val_images)
    print(f"✅ {cls}: जम्मा {len(images)} फोटो मध्ये -> Train मा: {remaining_train_count} वटा र Val मा: {len(val_images)} वटा सेट भयो।")

print("\n🎉 Work Finished! `ml_waste_system` photos are now split into Train and Val folders successfully.")