import time
import pandas as pd
import os
import Train_Test_models as ttm
from watchdog.observers.polling import PollingObserver as Observer
from watchdog.events import FileSystemEventHandler

class FileEventHandler(FileSystemEventHandler):
    def on_created(self, event):
        if event.is_directory:
            return
        if event.src_path.endswith('.csv'):
            print(f"New file detected: {event.src_path}")
            process_new_file(event.src_path)

old_csv = "./data/diamonds_1.csv"
def merge_csv_files(new_csv):
    old_data = pd.read_csv(old_csv)
    new_data = pd.read_csv(new_csv)
    merged_data = pd.concat([old_data, new_data])
    merged_data.to_csv(old_csv, index=False)
    # move the new csv file to a different folder
    new_data.to_csv(f"./data/trash/{new_csv.split('/')[-1]}", index=False)
    os.remove(new_csv)
    return merged_data

def process_new_file(file_path):
    new_data_preprocessed = []
    print(f"Processing file: {file_path}")
    new_data = merge_csv_files(file_path)
    new_data_preprocessed_tree = ttm.preprocess_input_for_tree_models(new_data)
    new_data_preprocessed_linear = ttm.preprocess_input_for_linear_models(new_data)
    new_data_preprocessed.append(new_data_preprocessed_linear)
    new_data_preprocessed.append(new_data_preprocessed_tree)
    
    ttm.fit_test_models(new_data_preprocessed)


if __name__ == "__main__":
    path = "./data/new_data"
    event_handler = FileEventHandler()
    observer = Observer()
    observer.schedule(event_handler, path, recursive=False)
    print(f"Starting observer for path: {path}")
    observer.start()
    observer.is_alive()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

