# Timbre Transfer Audio Demo

This page presents qualitative audio examples for six directed timbre transfer tasks using three timbres: **piano**, **guitar**, and **flute**.

Each row shows the same source melody transformed into a target timbre. The columns compare:

* **Source audio**: original input audio
* **LoRA result**: result from our Riffusion LoRA-based method
* **ISMIR2024 result**: result from the external reference model

---

## Audio Examples

<table>
  <thead>
    <tr>
      <th>Source Audio</th>
      <th>LoRA Result</th>
      <th>ISMIR2024 Result</th>
    </tr>
  </thead>

  <tbody>
    <tr>
      <td>
        <b>Piano → Guitar</b><br>
        <audio controls>
          <source src="assets/audio/piano_to_guitar/source.wav" type="audio/wav">
          Your browser does not support the audio element.
        </audio>
      </td>
      <td>
        <audio controls>
          <source src="assets/audio/piano_to_guitar/lora.wav" type="audio/wav">
          Your browser does not support the audio element.
        </audio>
      </td>
      <td>
        <audio controls>
          <source src="assets/audio/piano_to_guitar/ismir2024.wav" type="audio/wav">
          Your browser does not support the audio element.
        </audio>
      </td>
    </tr>

```
<tr>
  <td>
    <b>Piano → Flute</b><br>
    <audio controls>
      <source src="assets/audio/piano_to_flute/source.wav" type="audio/wav">
      Your browser does not support the audio element.
    </audio>
  </td>
  <td>
    <audio controls>
      <source src="assets/audio/piano_to_flute/lora.wav" type="audio/wav">
      Your browser does not support the audio element.
    </audio>
  </td>
  <td>
    <audio controls>
      <source src="assets/audio/piano_to_flute/ismir2024.wav" type="audio/wav">
      Your browser does not support the audio element.
    </audio>
  </td>
</tr>

<tr>
  <td>
    <b>Guitar → Piano</b><br>
    <audio controls>
      <source src="assets/audio/guitar_to_piano/source.wav" type="audio/wav">
      Your browser does not support the audio element.
    </audio>
  </td>
  <td>
    <audio controls>
      <source src="assets/audio/guitar_to_piano/lora.wav" type="audio/wav">
      Your browser does not support the audio element.
    </audio>
  </td>
  <td>
    <audio controls>
      <source src="assets/audio/guitar_to_piano/ismir2024.wav" type="audio/wav">
      Your browser does not support the audio element.
    </audio>
  </td>
</tr>

<tr>
  <td>
    <b>Guitar → Flute</b><br>
    <audio controls>
      <source src="assets/audio/guitar_to_flute/source.wav" type="audio/wav">
      Your browser does not support the audio element.
    </audio>
  </td>
  <td>
    <audio controls>
      <source src="assets/audio/guitar_to_flute/lora.wav" type="audio/wav">
      Your browser does not support the audio element.
    </audio>
  </td>
  <td>
    <audio controls>
      <source src="assets/audio/guitar_to_flute/ismir2024.wav" type="audio/wav">
      Your browser does not support the audio element.
    </audio>
  </td>
</tr>

<tr>
  <td>
    <b>Flute → Piano</b><br>
    <audio controls>
      <source src="assets/audio/flute_to_piano/source.wav" type="audio/wav">
      Your browser does not support the audio element.
    </audio>
  </td>
  <td>
    <audio controls>
      <source src="assets/audio/flute_to_piano/lora.wav" type="audio/wav">
      Your browser does not support the audio element.
    </audio>
  </td>
  <td>
    <audio controls>
      <source src="assets/audio/flute_to_piano/ismir2024.wav" type="audio/wav">
      Your browser does not support the audio element.
    </audio>
  </td>
</tr>

<tr>
  <td>
    <b>Flute → Guitar</b><br>
    <audio controls>
      <source src="assets/audio/flute_to_guitar/source.wav" type="audio/wav">
      Your browser does not support the audio element.
    </audio>
  </td>
  <td>
    <audio controls>
      <source src="assets/audio/flute_to_guitar/lora.wav" type="audio/wav">
      Your browser does not support the audio element.
    </audio>
  </td>
  <td>
    <audio controls>
      <source src="assets/audio/flute_to_guitar/ismir2024.wav" type="audio/wav">
      Your browser does not support the audio element.
    </audio>
  </td>
</tr>
```

  </tbody>
</table>

---

## Notes

The LoRA model is trained separately for each target timbre. During inference, the source audio spectrogram is edited toward the target timbre using the corresponding target-timbre LoRA checkpoint.

The ISMIR2024 result is used as an external reference baseline for qualitative comparison.
