import os
import torch
from kogpt2.pytorch_kogpt2 import get_pytorch_kogpt2_model
from gluonnlp.data import SentencepieceTokenizer
from kogpt2.utils import get_tokenizer
from http.server import BaseHTTPRequestHandler, HTTPServer

LIMIT = 500

tok_path = get_tokenizer()
model, vocab = get_pytorch_kogpt2_model()

def makeGPT(inputText) :
  tok = SentencepieceTokenizer(tok_path,  num_best=0, alpha=0)
  sent = inputText #입력받은 값으로 실행 
  toked = tok(sent)
  while 1:
    print(len(sent) - len(inputText))
    if len(sent) - len(inputText) > LIMIT:
      return (False, "")

    input_ids = torch.tensor([vocab[vocab.bos_token],]  + vocab[toked]).unsqueeze(0)
    pred = model(input_ids)[0]
    gen = vocab.to_tokens(torch.argmax(pred, axis=-1).squeeze().tolist())[-1]
    if gen == '</s>':
      break
    sent += gen.replace('▁', ' ')
    toked = tok(sent)
  return (True, sent)

class Handler (BaseHTTPRequestHandler):
  def do_POST(self):
    print("Length is " + self.headers.get('Content-Length'))
    content_len = int(self.headers.get('Content-Length'))
    post_body = self.rfile.read(content_len).decode('utf-8')

    success, resultText = makeGPT(post_body)
    print(post_body + " -> " + resultText)

    self.send_response(200)
    self.send_header("Content-type", "application/json; charset=UTF-8")
    self.end_headers()
    self.wfile.write(bytes("{\"success\": %s, \"result\": \"%s\"}" % (success and 'true' or 'false', resultText), "utf-8"))

addr = ('0.0.0.0', int(os.getenv('PORT', '80')))
httpd = HTTPServer(addr, Handler)
print("Server is now online")
httpd.serve_forever()
