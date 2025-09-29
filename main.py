from utils import *
import argparse
import os
import json
import time

def get_parser():
    parser = argparse.ArgumentParser(description='Manipulate the output of VLM using Adversarial Image')
    parser.add_argument('--model', default='llava', type=str, help='the victim model', choices=['llava', 'phi3', 'qwen', 'deepseek_vl'])
    parser.add_argument('--model_path', default='/data/model/llava-1.5-7b-hf', type=str, help='the path of model weights')
    parser.add_argument('--image_path', default='src.jpg', type=str, help='the path of source image')
    parser.add_argument('--data', default='./data.json', type=str, help='the prompt and target output')
    parser.add_argument('--epsilon', default=16/255, type=float, help='the upper bound of perturbation')
    parser.add_argument('--lr', default=1e-1, type=float, help='learning rate')
    parser.add_argument('--total_epoch', default=1000, type=int, help='the max number of epoch')
    parser.add_argument('--save_dir', default='results', type=str, help='save path')
    parser.add_argument('--eval', action='store_true', help='attack or evaluation')
    parser.add_argument('--min_loss', default=0.1, type=float, help='the min loss for termination')
    parser.add_argument('--task', default='jailbreaking', type=str, choices=['jailbreaking', 'hijacking', 'privacy breaches', 'Denial_of_Service', 'sponge_examples', 'watermarking'])
    parser.add_argument('--exact_threhold', default=0.5, type=float, help='the threshold of loss for starting the exact match')
    return parser.parse_args()

def attack(image, prompt, target_output, model, processor, model_name, epsilon=None, lr=1e-1, total_epoch=1000, save_path='results/adv_best.png', min_loss=0.1, exact_threhold=0.5):
    inputs, conv_prompt, target_slice, loss_slice = get_inputs(processor, prompt, image, target_output, model_name)
      
    to_tensor = transforms.ToTensor()
    
    src_image = to_tensor(np.array(image)).unsqueeze(0).cuda()
    upper_bound, lower_bound = None, None
    if epsilon is not None:
        upper_bound = torch.clamp(src_image+epsilon, 0, 1)
        lower_bound = torch.clamp(src_image-epsilon, 0, 1)
    
    inputs = inputs.to('cuda')
    
    if model_name == 'llava':
        preprocess = Llava_Process(processor, upper_bound, lower_bound)
    elif model_name == 'phi3':
        src_image = src_image.squeeze(0)
        upper_bound = upper_bound.squeeze(0)
        lower_bound = lower_bound.squeeze(0)
        preprocess = Phi3_Process(processor, upper_bound, lower_bound)
    elif model_name == 'qwen':
        preprocess = Qwen_Process(processor, upper_bound, lower_bound)
    elif model_name == 'deepseek_vl':
        src_image = src_image.squeeze(0)
        upper_bound = upper_bound.squeeze(0)
        lower_bound = lower_bound.squeeze(0)
        preprocess = DeepSeek_Process(processor, upper_bound, lower_bound)
    else:
        raise NotImplementedError(f"Unsupported model_name = {model_name}")
    
    src_image = reverse_sigmoid(src_image)
    src_image.requires_grad = True

    crit = nn.CrossEntropyLoss(reduction='none')
    optimizer = torch.optim.Adam([src_image], lr=lr)
    best_loss = float('inf')
    if args.task != 'sponge_examples':
        for epoch in range(total_epoch):
            optimizer.zero_grad()
            
            if model_name == 'deepseek_vl':
                prepare = preprocess(src_image, conv_prompt)
                inputs_embeds = model.prepare_inputs_embeds(**prepare)
                logits = model.language_model(inputs_embeds = inputs_embeds).logits
            else:
                inputs, conv_prompt, target_slice, loss_slice = get_inputs(processor, prompt, image, target_output, model_name)
                
                model_kwargs = preprocess(images=src_image, inputs=inputs, model=model)
                logits = model(**model_kwargs).logits

            loss = crit(logits[:, loss_slice, :].transpose(1, 2), inputs.input_ids[:, target_slice]).mean(dim=-1)

            print(f'{epoch}/{total_epoch}: best loss={best_loss:.3f}, current loss={loss.cpu().detach().numpy()[0]:.3f}')
            
            if loss.cpu().detach().numpy()[0] < best_loss:
                best_loss = loss.cpu().detach().numpy()[0]
                save_image(preprocess.convert_to_image(src_image), save_path)
                if best_loss < min_loss:
                    break
            loss.backward()
            optimizer.step()

    else:
        target_token = 2 # Please chose the EOS_TOKEN ID for different VLMs.

        best_loss = float('inf')
        max_token = 0
        flag = 0
        for epoch in range(total_epoch):
            optimizer.zero_grad()
            
            inputs, conv_prompt, target_slice, loss_slice = get_inputs(processor, prompt, image, target_output, model_name)
            
            model_kwargs = preprocess(images=src_image, inputs=inputs, model=model)

            with torch.no_grad():
                start_time = time.time()
                generate_ids = model.generate(**model_kwargs, max_new_tokens=10000, do_sample=False)
                end_time = time.time()
                cost_time = end_time - start_time
                curr_token_len = generate_ids.shape[1]
                response = processor.batch_decode(generate_ids, skip_special_tokens=True)

            model_kwargs["input_ids"] = generate_ids

            
            logits = model(**model_kwargs).logits
            
            loss = logits.softmax(dim=-1)[0, target_slice.stop:-1, target_token].sum()

            print(f'{epoch}/{total_epoch}: best loss={best_loss:.3f}, current loss={loss.cpu().detach().numpy():.3f}, max_token={curr_token_len}, max_token_len={max_token}, cost_times={cost_time}')
            
            if curr_token_len > max_token:
                max_token = curr_token_len
                save_image(preprocess.convert_to_image(src_image), save_path)
                res = {"response":response, "token_len":curr_token_len, "cost_time":cost_time}
                if max_token >= target_slice.stop + 10000:
                    break

            loss.backward()
            optimizer.step()


def infer(image, prompt, model, processor, model_name):
    inputs, conv_prompt, target_slice, loss_slice = get_inputs(processor, prompt, image, "", model_name)
    inputs = inputs.to('cuda')
    if model_name == 'llava':
        generate_ids = model.generate(**inputs, max_new_tokens=300, do_sample=False)
        response = processor.batch_decode(generate_ids, skip_special_tokens=True)
    elif model_name == 'phi3':
        generation_args = { 
            "max_new_tokens": 300, 
            "temperature": 0.0, 
            "do_sample": False, 
        } 

        generate_ids = model.generate(**inputs, eos_token_id=processor.tokenizer.eos_token_id, **generation_args) 

        # remove input tokens 
        generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
        response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0] 
    elif model_name == 'qwen':
        generated_ids = model.generate(**inputs, max_new_tokens=300)

        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        response = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
    elif model_name=='deepseek_vl':
        inputs_embeds = model.prepare_inputs_embeds(**inputs)
        outputs = model.language_model.generate(
            inputs_embeds = inputs_embeds,
            attention_mask = inputs.attention_mask,
            pad_token_id = processor.tokenizer.eos_token_id,
            bos_token_id = processor.tokenizer.bos_token_id,
            eos_token_id = processor.tokenizer.eos_token_id,
            max_new_tokens = 100,
            do_sample = False,
            use_cache=True
        )
        response = processor.tokenizer.decode(outputs[0].cpu().tolist(),skip_special_tokens=True)

    return response


if __name__ == '__main__':
    args = get_parser()
    image = Image.open(args.image_path)
    with open(args.data, 'r') as file:
        data = json.load(file)
    
    save_dir = os.path.join(args.save_dir, args.model)    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    model, processor = load_model_and_processor(args.model, args.model_path)
    
    for i, item in enumerate(data):
        prompt = item['prompt']
        target_output = item['target_output']
        if args.eval:
            print(f"--------------------------------Start {i}-th inference!-----------------------------------\nprompt={prompt}")
            image = Image.open(os.path.join(save_dir, f'{i}.png'))
            response = infer(image, prompt, model, processor, args.model)
            print(response)
        else:
            if args.model == 'deepseek_vl':
                image = image.resize((1024,1024))
            print(f"--------------------------------Start {i}-th attack!-----------------------------------\nprompt={prompt}\ntarget_output={target_output}")
            attack(image, prompt, target_output, model, processor, args.model, args.epsilon, args.lr, args.total_epoch, os.path.join(save_dir, f'{i}.png'), args.min_loss, args.exact_threhold)