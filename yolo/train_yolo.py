from ultralytics import YOLO
import multiprocessing as mp

def main():
    model = YOLO("yolo11s.pt")

    model.train(
        data=r"D:\A.I\data.yaml",
        epochs=200,
        imgsz=720,
        batch=-1,
        device=0,
        workers=4,
        patience=30,
        cos_lr=True,
        lr0=0.01, lrf=0.01,
        mosaic=0.3, mixup=0.1,
        cache='disk'
    )

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)  # khuyến nghị trên Windows
    main()



