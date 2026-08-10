import os
import shutil
import random

# Source र Destination paths
base_dir = 'ml_waste_system'
valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

# 'invalid_or_clean' पनि थपिएको ४ वटा क्याटगोरीहरू
classes = ['high_waste', 'invalid_or_clean', 'low_waste', 'medium_waste']
split_ratio = 0.8  # 80% Train, 20% Val

print("🔄 `ml_waste_system/train` बाट 80/20 अनुपातमा डाटा विभाजन हुँदैछ...\n")

for cls in classes:
    train_cls_dir = os.path.join(base_dir, 'train', cls)
    val_cls_dir = os.path.join(base_dir, 'val', cls)

    # Val फोल्डर छैन भने बनाउने
    os.makedirs(val_cls_dir, exist_ok=True)

    if not os.path.exists(train_cls_dir):
        print(f"⚠️ Warning: {train_cls_dir} फोल्डर भेटिएन!")
        continue

    # Train फोल्डर भित्र भएका सबै फोटोहरूको लिस्ट लिने
    images = [
        f for f in os.listdir(train_cls_dir)
        if os.path.isfile(os.path.join(train_cls_dir, f)) and os.path.splitext(f)[1].lower() in valid_extensions
    ]

    if len(images) == 0:
        print(f"⚠️ {cls} भित्र कुनै फोटोहरू छैनन्!")
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
    print(f"✅ {cls}: कुल {len(images)} -> Train: {remaining_train_count} | Val: {len(val_images)}")

print("\n🎉 काम सकियो! `ml_waste_system` का फोटोहरू सफलतापूर्वक Train र Val फोल्डरमा विभाजन भए।")