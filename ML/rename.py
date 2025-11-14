import os

directory = r"C:\Users\Sandy\OneDrive - Högskolan Väst\Semester 3 Quarter 1\SYI700\2 Project\Code\SYI_Scripts\image_data\Classifier\validation\bot_mix"

for filename in os.listdir(directory):
    src = os.path.join(directory, filename)
    if os.path.isfile(src):
        new_filename = filename.replace("top", "bot")
        dst = os.path.join(directory, new_filename)
        os.rename(src, dst)
        print(f"Renamed: {filename} -> {new_filename}")