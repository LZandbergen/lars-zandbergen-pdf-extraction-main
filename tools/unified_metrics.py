import weave
import re

NN = 'nn'
NV = 'nv'
VN = 'vn'
VC = 'vc'
VW = 'vw'

def clean_string(text: str):
    """
    Cleans a string by replacing all non-lowercase alphanumeric characters
    with a single whitespace and ensuring no consecutive whitespaces.

    Args:
        text: The input string.

    Returns:
        The cleaned string.
    """
    cleaned_text = re.sub(r'[^a-z0-9]+', ' ', text.lower())
    cleaned_text = cleaned_text.strip()
    cleaned_text = re.sub(r'\s{2,}', ' ', cleaned_text)
    return cleaned_text

def parse_attributes(text: str, logger):
    logger.debug(f"Running parse_attributes on text: {text}")
    """
    Convert a decoded model output string into a dictionary of attributes.
    
    It assumes that the text is in a newline-delimited "key : value" format.
    """
    text = text.strip()
    attr_dict = {}
    for line in text.splitlines():
        if ':' in line:
            key, value = line.split(':', 1)
            cleaned_key = clean_string(key)
            cleaned_value = clean_string(value)
            attr_dict[cleaned_key] = cleaned_value
    return attr_dict

@weave.op()
def evaluate_response(label_dict, response_lines, logger):
    """
    Evaluates to what category each expected label belongs to, works with multiple attributes

    Args:
        label_dict (dict): Contains the desired labels and their value as a key-value pair.
        response_lines (list): Contains each seperated string from the LLM response formatted as "attribute : value"
        logger (logger): Created function which contains the correct logging file.
    
    Categories:
      - NN: No predicted value when ground truth is "none" (not used in calculating metrics).
      - NV: Predicted value when ground truth is "none". (probably really rare)
      - VN: No predicted value when ground truth has a value.
      - VC: Correct predicted value (matches ground truth).
      - VW: Predicted value exists but does not match ground truth.
      
    Returns:
        dict: a dictionary which contains the counts of each evaluation metric.
    """
    eval_dict = {
        NN: 0,
        NV: 0,
        VN: 0,
        VC: 0,
        VW: 0
    }
    
    for lbl_key, lbl_value in label_dict.items():
        # Extract predicted value for this key (assumes each line has "key : value" format)

        matching_lines = [line for line in response_lines if line.strip().lower().startswith(lbl_key.lower())]
        if matching_lines:
            # Only evaluate first matching line.
            pred_value = matching_lines[0].split(":", 1)[-1].strip(' "').lower()
        else:
            pred_value = None

        # Determine category
        # When ground truth indicates no attribute (using "none" as a marker)
        if lbl_value.lower() == "none":
            if pred_value is None or pred_value == "none" or pred_value == r"n/a":
                # NN: Correctly predicted no value (not counted in metrics)
                eval_dict[NN] += 1
                result = NN
            else:
                # NV: Predicted a value when there should be none.
                eval_dict[NV] += 1
                result = NV
        else:
            # Ground truth has a valid attribute value.
            if pred_value is None or pred_value == "none" or pred_value == r"n/a":
                # VN: Missed the attribute.
                eval_dict[VN] += 1
                result = VN
            elif pred_value == lbl_value.lower():
                # VC: Correct prediction.
                eval_dict[VC] += 1
                result = VC
            else:
                # VW: Predicted a wrong value.
                eval_dict[VW] += 1
                result = VW

        logger.info(f"Key: {lbl_key} | Predicted value: {pred_value} | Actual value: {lbl_value} | ({result})")

    logger.info(f"Evaluation -> VC: {eval_dict[VC]}, VN: {eval_dict[VN]}, VW: {eval_dict[VW]}, NV: {eval_dict[NV]}, NN: {eval_dict[NN]}")

    return eval_dict

@weave.op()
def calculate_metrics(eval_dict):
    """
    This function calculates the precision, recall and F1 score based on the evaluation metrics.

    Args:
        eval_dict (dict): A dictionary containing the counts of each evaluation metric.

    Returns:
        tuple: A tuple containing precision, recall, and f1 scores.
    """
    precision = round((eval_dict[VC] / (eval_dict[NV] + eval_dict[VC] + eval_dict[VW])) * 100, 2) if (eval_dict[NV] + eval_dict[VC] + eval_dict[VW]) > 0 else 0
    recall = round((eval_dict[VC] / (eval_dict[VN] + eval_dict[VC] + eval_dict[VW])) * 100, 2) if (eval_dict[VN] + eval_dict[VC] + eval_dict[VW]) > 0 else 0
    f1 = round(2 * precision * recall / (precision + recall), 2) if (precision + recall) > 0 else 0

    return precision, recall, f1
    