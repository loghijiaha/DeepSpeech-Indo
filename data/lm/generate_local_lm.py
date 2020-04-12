import gzip
import io
import os
import subprocess
import tempfile
import bz2

from collections import Counter
from urllib import request

def main():

  with tempfile.TemporaryDirectory() as tmp:
    data_upper = '/content/DeepSpeech-Indo/data/lm/output'
    

    # Convert to lowercase and count word occurences.
    counter = Counter()
    data_lower = os.path.join(tmp, 'lower.txt')
    print('Converting to lower case and counting word frequencies...')
    with open(data_lower, 'w', encoding='utf-8') as lower:
      with open(data_upper, encoding='utf8') as upper:
        for line in upper:
          line_lower = line.lower()
          counter.update(line_lower.split())
          lower.write(line_lower)

    # Build pruned LM.
    lm_path = '/content/DeepSpeech-Indo/data/lm/lm.arpa'
    print('Creating ARPA file...')
    subprocess.check_call([
      '/content/bin/lmplz', '--order', '5',
               '--temp_prefix', tmp,
               '--memory', '50%',
               '--text', data_lower,
               '--arpa', lm_path,
               '--prune', '0', '0', '1'
    ])

    vocab_str = '\n'.join(word for word, count in counter.most_common(1197913))
    with open('/content/DeepSpeech-Indo/data/lm/vocabulary.txt', 'w') as fout:
     	
      fout.write(vocab_str)

    # Filter LM using vocabulary of top 500k words
    print('Filtering ARPA file...')
    filtered_path = os.path.join(tmp, 'lm_filtered.arpa')
    subprocess.run(['/content/bin/filter', 'single', 'model:{}'.format(lm_path), filtered_path], input=vocab_str.encode('utf-8'), check=True)

    # Quantize and produce trie binary.
    print('Building lm.binary...')
    subprocess.check_call([
      '/content/bin/build_binary', '-a', '255',
                      '-q', '8',
                      '-v',
                      'trie',
                      filtered_path,
                      '/content/DeepSpeech-Indo/data/lm/lm.binary'
    ])

if __name__ == '__main__':
  main()
