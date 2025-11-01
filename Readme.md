Yolo Automartic license plate dtection model.

An end-to-end **License Plate Detection** web application built with **Streamlit**, **YOLOv8**, and **EasyOCR**.  
This app can detect car license plates in both **images** and **videos**, display detection confidence, extract plate numbers, and save results automatically.

## 📸 Features

✅ **Upload Images or Videos** — Supports `.jpg`, `.jpeg`, `.png`, `.mp4`, `.mov`, `.avi`  
✅ **YOLOv8 Model Integration** — Pre-trained on license plate dataset for fast detection  
✅ **EasyOCR** — Reads alphanumeric license plate text  
✅ **Confidence Display** — Shows model confidence (%) above each bounding box  
✅ **Auto-save Outputs** — All predictions are saved to the `output/` directory  
✅ **Professional Streamlit UI/UX** — Clean, responsive layout with progress tracking  
✅ **Download Option** — Easily download processed results (image or video)  

## 🧠 Tech Stack

- [Streamlit](https://streamlit.io/) — for frontend UI  
- [YOLOv8 (Ultralytics)](https://github.com/ultralytics/ultralytics) — for object detection  
- [OpenCV](https://opencv.org/) — for image/video frame processing  
- [EasyOCR](https://github.com/JaidedAI/EasyOCR) — for optical character recognition  
- [Python 3.8+](https://www.python.org/)
