"""Train a J/Z sequence network. Split source clips before any augmentation."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from .motion_features import MOTION_FEATURE_VERSION, motion_features

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=ROOT / "datasets" / "motion.json")
    parser.add_argument("--output", type=Path, default=ROOT / "signspeak" / "static" / "models")
    args = parser.parse_args()
    unique = {}
    for record in json.loads(args.dataset.read_text(encoding="utf-8"))["samples"]:
        vector = motion_features(record["frames"])
        fingerprint = hashlib.sha256(vector.round(5).tobytes()).hexdigest()
        if fingerprint not in unique:
            unique[fingerprint] = dict(record, features=vector.tolist(), fingerprint=fingerprint)
    records = list(unique.values())
    x = np.array([r["features"] for r in records])
    y = np.array([r["letter"] for r in records])
    if set(y) != {"J", "Z"} or min((y == label).sum() for label in ["J", "Z"]) < 5:
        raise ValueError("At least five unique complete clips per letter are required")
    groups = [r.get("participant") for r in records]
    ids = np.arange(len(records))
    if all(groups) and len(set(groups)) >= 3:
        trainval, test = next(GroupShuffleSplit(n_splits=1, test_size=.2, random_state=42).split(ids, y, groups))
        ti, vi = next(GroupShuffleSplit(n_splits=1, test_size=.25, random_state=43).split(trainval, y[trainval], np.array(groups)[trainval]))
        train, val = trainval[ti], trainval[vi]
        split = "participant-disjoint"
    else:
        trainval, test = train_test_split(ids, test_size=.2, random_state=42, stratify=y)
        train, val = train_test_split(trainval, test_size=.25, random_state=43, stratify=y[trainval])
        split = "source-clip-disjoint; participant IDs unavailable or incomplete"
    if any(set(y[indices]) != {"J", "Z"} for indices in [train, val, test]):
        raise ValueError("Every split needs both letters. Collect more participants/clips.")
    scaler = StandardScaler().fit(x[train])
    rng = np.random.default_rng(17)
    augmented = np.concatenate([x[train]]+[x[train]+rng.normal(0,.01,x[train].shape) for _ in range(4)])
    classifier = MLPClassifier(hidden_layer_sizes=(48,24), solver="lbfgs", alpha=.2, random_state=17, max_iter=1500)
    classifier.fit(scaler.transform(augmented), np.tile(y[train],5))
    probability = classifier.predict_proba(scaler.transform(x[test]))
    labels = classifier.classes_.tolist()
    prediction = np.array(labels)[probability.argmax(axis=1)]
    support = {label: x[train][y[train] == label].tolist() for label in labels}
    def distance(features,label):
        return np.sqrt(((features[:,None,:]-np.array(support[label])[None,:,:])**2).mean(axis=2)).min(axis=1)
    gates = {label:float(max(.25,np.quantile(distance(x[val][y[val] == label],label),.95))) for label in labels}
    accepted = [str(label) if probability[i].max() >= .85 and distance(x[test][i:i+1],label)[0] <= gates[label] else "_" for i,label in enumerate(prediction)]
    kernels = [layer.copy() for layer in classifier.coefs_]
    biases = [layer.copy() for layer in classifier.intercepts_]
    # Convert sklearn's binary sigmoid logit to equivalent [0, logit] softmax.
    kernels[-1] = np.column_stack([np.zeros(kernels[-1].shape[0]), kernels[-1][:,0]])
    biases[-1] = np.array([0,biases[-1][0]])
    args.output.mkdir(parents=True,exist_ok=True)
    def save(name,data):
        (args.output/name).write_text(json.dumps(data,separators=(",",":"),allow_nan=False),encoding="utf-8")
    save("motion-classifier.json",{"feature_version":MOTION_FEATURE_VERSION,"labels":labels,"mean":scaler.mean_.tolist(),"scale":scaler.scale_.tolist(),"layers":[{"kernel":k.tolist(),"bias":b.tolist()} for k,b in zip(kernels,biases)],"support":support,"distance_thresholds":gates})
    save("motion-evaluation.json",{"dataset_sha256":hashlib.sha256(args.dataset.read_bytes()).hexdigest(),"split":split,"classes":labels,"counts":{"unique_total":len(records),"train":len(train),"validation":len(val),"test":len(test)},"class_counts":{label:int((y == label).sum()) for label in labels},"raw_classifier_accuracy":float(accuracy_score(y[test],prediction)),"gated_accuracy_at_threshold_085":float(accuracy_score(y[test],accepted)),"classification_report":classification_report(y[test],prediction,output_dict=True,zero_division=0),"confusion_matrix":confusion_matrix(y[test],prediction,labels=labels).tolist(),"test_examples":[{"frames":records[i]["frames"],"features":x[i].tolist(),"expected":str(y[i]),"probabilities":probability[j].tolist()} for j,i in enumerate(test)],"split_clips":{key:[records[i].get("clip",records[i]["fingerprint"]) for i in indices] for key,indices in [("train",train),("validation",val),("test",test)]},"warning":"Experimental movement model; small clip subset; signer independence unknown. No held-out unsupported movements, so unknown-motion rejection is unmeasured."})
    save("motion-metadata.json",{"ready":True,"labels":labels,"feature_version":MOTION_FEATURE_VERSION,"training_samples":len(train),"test_samples":len(test),"test_accuracy":float(accuracy_score(y[test],prediction)),"split":split,"experimental":True,"source":"SigNN J/Z public video subset, CC0","recording_ms":2400})
    refs = {}
    for label in labels:
        candidates = [i for i in train if y[i] == label]
        center = np.median(x[candidates],axis=0)
        best = min(candidates,key=lambda i:np.linalg.norm(x[i]-center))
        refs[label] = records[best]["frames"]
    save("motion-references.json",refs)
    print(json.dumps({"train":len(train),"validation":len(val),"test":len(test),"test_accuracy":float(accuracy_score(y[test],prediction)),"gated_accuracy":float(accuracy_score(y[test],accepted))},indent=2))


if __name__ == "__main__":
    main()
