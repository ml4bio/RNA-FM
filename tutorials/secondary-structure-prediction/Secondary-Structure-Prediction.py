import fm
import torch



if __name__=="__main__":
    #path= "/data/home/chenjiayang/projects/RNA-FM/redevelop/pretrained/Models/SS/RNA-FM-ResNet_PDB-All.pth"
    model, alphabet = fm.downstream.build_rnafm_resnet(type="ss") #, model_location=path)
    batch_converter = alphabet.get_batch_converter()
    model.eval()  # disables dropout for deterministic results

    # Prepare data
    data = [
        ("RNA1", "GGGUGCGAUCAUACCAGCACUAAUGCCCUCCUGGGAAGUCCUCGUGUUGCACCCCU"),
    ]
    batch_labels, batch_strs, batch_tokens = batch_converter(data)

    input = {
        "description": batch_labels,
        "token": batch_tokens
    }

    # Secodnary Structure Prediction (on CPU)
    with torch.no_grad():
        results = model(input)
    ss_prob_map = results["r-ss"]
    print(ss_prob_map)
    print(ss_prob_map.shape)





