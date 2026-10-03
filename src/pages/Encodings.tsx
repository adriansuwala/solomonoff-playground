/**
 * Encodings (Appendix B): what the "bytes" actually are.
 *
 * This matters for reading every other page. The paper evaluates on seven
 * modalities through one interface — next-byte prediction over raw bytes — and a
 * reader who has not internalised that the DNA benchmark is an 8-symbol alphabet
 * inside a 256-way softmax will misread the loss values. In particular: a
 * predictor that knows only eight values are possible gets 3 bits/byte on DNA,
 * while uniform over the full byte space costs 8.
 */
import { useState } from 'react'
import { loadEncodings } from '@/data/loaders'
import { Async, Card, Claim, Disclosure, PaperRef } from '@/components/UI'

const CORPORA: Record<string, { label: string; ref: 'appB' | 'fig8' | 'fig9' | 'fig10'; blurb: string }> = {
  dclm_ranked: {
    label: 'Natural text — DCLM',
    ref: 'appB',
    blurb:
      'UTF-8 bytes of filtered Common Crawl text, concatenated within each shard ' +
      'and cut into fixed windows. The model must exploit spelling, punctuation, ' +
      'word structure and local syntax through the same byte interface as everything else.',
  },
  dna: {
    label: 'DNA — 8 symbols',
    ref: 'appB',
    blurb:
      'The first 32 MiB of the KoLMogorov release, derived from the GRCh38 human ' +
      'reference. Values 0–7 map to a, c, t, g, A, C, T, G — lowercase is soft ' +
      'masking for repetitive regions. Records of 255 symbols, no overlap.',
  },
  audio_8bit: {
    label: 'Audio — 8-bit PCM',
    ref: 'appB',
    blurb:
      'Speech Commands one-second mono recordings with the WAV header discarded. ' +
      'Each signed 16-bit sample is mapped to floor((s + 32768) / 256), so silence ' +
      'becomes 128 and neighbouring bytes stay neighbouring in time.',
  },
  audio_16bit: {
    label: 'Audio — 16-bit PCM',
    ref: 'appB',
    blurb:
      'The same recordings without the 8-bit quantisation step, preserving more ' +
      'of the waveform at the cost of far more bytes per second of audio.',
  },
}

const ASCII: Record<number, string> = {
  9: '\\t', 10: '\\n', 13: '\\r', 32: '·', 127: '␡',
}

export function Encodings() {
  const [corpus, setCorpus] = useState<string>('dclm_ranked')
  const [seqIndex, setSeqIndex] = useState(0)
  const [hover, setHover] = useState<number | null>(null)

  return (
    <>
      <h1 className="page__title">What the bytes are</h1>
      <p className="page__lede">
        Every claim about "bits per byte" in this app rests on one interface: the
        learner predicts the next byte of a raw stream, over a vocabulary of all
        256 values, for every modality. Seven very different kinds of data are
        funnelled through that single interface — which is what makes the transfer
        result meaningful, and what makes the absolute loss numbers hard to
        compare naively.
      </p>

      <h2 className="section">The interface</h2>
      <Card title="One alphabet, seven modalities">
        <div className="kv">
          <span className="kv__k">vocabulary</span>
          <span className="kv__v">all 256 byte values</span>
          <span className="kv__k">context</span>
          <span className="kv__v">4096 tokens</span>
          <span className="kv__k">task</span>
          <span className="kv__v">next-byte prediction, cross-entropy</span>
          <span className="kv__k">metric</span>
          <span className="kv__v">bits per byte (lower is better)</span>
        </div>
        <p className="body" style={{ marginTop: 12 }}>
          Programs are prefixed with the byte <code>S</code> and their outputs with{' '}
          <code>O</code>, so one model architecture discriminates the two kinds of
          stream from a single byte.{' '}
          <PaperRef reference="sec2.1" inline />
        </p>
      </Card>

      <div className="claim claim--caveat">
        <p className="claim__text">
          <strong>Losses are not comparable across modalities.</strong> A uniform
          predictor over 256 byte values costs exactly 8 bits/byte on everything,
          so 5 bits/byte on text and 5 bits/byte on DNA mean very different things.
          A predictor that knows DNA only uses eight symbols gets 3 bits/byte
          before learning anything; the same for text is impossible. Compare
          exponents and frontiers within a modality, never across them.{' '}
          <PaperRef reference="appB" inline />
        </p>
      </div>

      <h2 className="section">Look at the actual bytes</h2>
      <Async load={loadEncodings}>
        {(data) => {
          const available = Object.keys(data.samples).filter(
            (k) => CORPORA[k] && data.samples[k]!.length > 0,
          )
          const chosen = available.includes(corpus) ? corpus : available[0]!
          const seqs = data.samples[chosen] ?? []
          const seq = seqs[Math.min(seqIndex, seqs.length - 1)] ?? []
          const info = CORPORA[chosen]!
          return (
            <>
              <div className="controls">
                <div className="field">
                  <label className="field__label" htmlFor="enc-corpus">Corpus</label>
                  <select
                    id="enc-corpus"
                    value={chosen}
                    onChange={(e) => { setCorpus(e.target.value); setSeqIndex(0) }}
                  >
                    {available.map((k) => (
                      <option key={k} value={k}>{CORPORA[k]!.label}</option>
                    ))}
                  </select>
                </div>
                <div className="field">
                  <label className="field__label" htmlFor="enc-seq">Record</label>
                  <select
                    id="enc-seq"
                    value={seqIndex}
                    onChange={(e) => setSeqIndex(Number(e.target.value))}
                  >
                    {seqs.map((_, i) => (
                      <option key={i} value={i}>{i + 1}</option>
                    ))}
                  </select>
                </div>
              </div>

              <Card
                title={info.label}
                note={<>{info.blurb} <PaperRef reference={info.ref} inline /></>}
              >
                <div className="tape" onMouseLeave={() => setHover(null)}>
                  {seq.map((b, i) => {
                    const distinct = new Set(seq).size
                    const ascii = ASCII[b] ?? (
                      b >= 32 && b < 127 ? String.fromCharCode(b) : ''
                    )
                    return (
                      <div
                        key={i}
                        className={[
                          'tape__cell',
                          b === 0 ? 'tape__cell--zero' : '',
                          hover === i ? 'tape__cell--head' : '',
                        ].join(' ')}
                        style={{
                          opacity: distinct <= 8 ? 1 : 0.55 + (b / 255) * 0.45,
                        }}
                        title={`position ${i}: value ${b}${ascii ? ` (${ascii})` : ''}`}
                        onMouseEnter={() => setHover(i)}
                      >
                        {ascii || b}
                      </div>
                    )
                  })}
                </div>
                <p className="card__note" style={{ marginTop: 10 }}>
                  {seq.length} bytes shown of a {data.contextLength}-token context.{' '}
                  {hover !== null && seq[hover] !== undefined && (
                    <>
                      Position {hover}: value {seq[hover]}
                      {ASCII[seq[hover]!] ? `, shown as ${ASCII[seq[hover]!]}` : ''}.
                    </>
                  )}
                </p>
                {chosen === 'dclm_ranked' && (
                  <Disclosure summary="Read as text (Figures 8 and 10 show the paper's own illustration)">
                    <pre className="program" style={{ whiteSpace: 'pre-wrap' }}>
                      {seq.map((b) =>
                        b >= 32 && b < 127 ? String.fromCharCode(b) : '·',
                      ).join('')}
                    </pre>
                  </Disclosure>
                )}
                {chosen === 'dna' && (
                  <p className="card__note">
                    Values 0–7 map to a, c, t, g, A, C, T, G in that order. The
                    repeated runs visible above are soft-masked low-complexity
                    regions, which is the structure the paper's DNA exponent
                    (0.435, far above every other modality) reflects.
                  </p>
                )}
              </Card>
            </>
          )
        }}
      </Async>

      <h2 className="section">Why one interface?</h2>
      <Claim reference="sec1" also={['appB']}>
        Routing every modality through the same byte interface is what makes the
        transfer test clean. Because the learner never receives a dataset-specific
        tokeniser, vocabulary or input format, any improvement on natural data has
        to come from structure it learned from programs — not from engineering
        particular to one corpus.
      </Claim>

      <Card title="The costs of that choice, stated plainly">
        <ul className="body" style={{ paddingLeft: 18, margin: 0 }}>
          <li>
            <strong>Audio is brutally byte-expensive.</strong> 16 kHz 8-bit PCM is
            16,000 bytes per second, so a 4096-token context spans about a quarter
            of a second. That is why the paper also evaluates symbolic music, where
            four bytes per quarter note pack whole phrases into the context.
          </li>
          <li>
            <strong>Melody is quantised hard.</strong> The Mutopia encoding is a
            16th-note grid: bytes 0–127 are absolute MIDI pitch, 128 holds the
            previous note, 129 is a pause, 130 ends the piece. Everything is
            monophonic by construction — one active note per grid cell.
          </li>
          <li>
            <strong>CIFAR-10 is checked both ways.</strong> Planar (CHW) and
            interleaved (HWC) encodings of identical images, to confirm the 1-D
            ordering does not matter. The paper reports no appreciable effect.
          </li>
        </ul>
      </Card>

      <h2 className="section">Datasets not shown here</h2>
      <Card
        title="Present in the scaling data, not baked into this bundle"
        note="The four corpora above ship pre-baked with the authors' scoring code. The rest are generated by their prepare_* scripts, which download from public sources."
      >
        <p className="body" style={{ margin: 0 }}>
          Metamath <code>set.mm</code>, AITDCC C source, GitHub Python, Mutopia
          melody, MusicNet, ESC-50, CIFAR-10, and the binary corpora (glibc rand,
          ATLAS float32, astronomy). Every one of them appears in the scaling
          explorer, because that data ships pre-scored.{' '}
          <PaperRef reference="appB" inline />
        </p>
      </Card>
    </>
  )
}