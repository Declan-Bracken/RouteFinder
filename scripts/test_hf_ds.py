from datasets import load_dataset                                                                                                                                                                       
import os
                                                                                                                                                                                                        
ds = load_dataset("DeclanBracken/RouteFinderDatasetV2", token=os.environ["HF_TOKEN"])
labels = ds["train"]["label"]                                                                                                                                                                           
print(f"Min: {min(labels)}, Max: {max(labels)}, Unique: {len(set(labels))}")                                                                                                                            
print(f"Sample labels: {sorted(set(labels))[:10]}")
