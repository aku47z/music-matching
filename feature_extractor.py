"""
Feature Extractor Module
Handles MIDI parsing and feature extraction for plagiarism detection.
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional
from music21 import converter, stream, note, meter, chord

@dataclass
class NoteFeature:
    """Represents extracted features for a single note."""
    pitch_interval: float  # Δp: relative pitch in semitones
    duration_ratio: float  # Δd: relative rhythmic ratio
    is_downbeat: int       # 1 if on strong beat, 0 otherwise
    
    def to_tuple(self) -> Tuple[float, float, int]:
        return (self.pitch_interval, self.duration_ratio, self.is_downbeat)


def parse_midi(filepath: str, quantize: bool = True) -> List[note.Note]:
    """
    Parse a MIDI file and extract monophonic melody notes.
    
    Args:
        filepath: Path to the MIDI file
        quantize: Whether to quantize notes to nearest 16th note (reduces timing noise)
        
    Returns:
        List of music21 Note objects representing the melody
    """
    score = converter.parse(filepath)
    
    # Temporal Quantization: Snap notes to nearest 16th note
    if quantize:
        score = score.quantize(quarterLengthDivisors=(4,), processOffsets=True, processDurations=True)
    
    # Create measures for proper beat detection
    try:
        score = score.makeMeasures(inPlace=False)
    except:
        pass  # Some MIDI files may not support measure creation
    
    # Flatten to get all notes (handles multi-part scores)
    notes_list = []
    
    # Filter out non-note events (only keep actual Notes)
    for element in score.flatten().notesAndRests:
        if isinstance(element, note.Note):
            notes_list.append(element)
        elif isinstance(element, chord.Chord):
            # For chords, take the highest note (likely melody)
            highest = element.sortAscending()[-1]
            n = note.Note(highest.pitch)
            n.offset = element.offset
            n.quarterLength = element.quarterLength
            notes_list.append(n)
        # Skip rests and other events (sustain pedal, pitch bend, etc.)
    
    # Sort by offset to ensure chronological order
    notes_list.sort(key=lambda n: n.offset)
    
    # Remove duplicate notes at same offset (keep first/highest)
    cleaned = []
    last_offset = -1
    for n in notes_list:
        if n.offset != last_offset:
            cleaned.append(n)
            last_offset = n.offset
    
    return cleaned


def is_downbeat(note_obj: note.Note, time_signature: Optional[meter.TimeSignature] = None) -> int:
    """
    Determine if a note falls on a strong beat (downbeat) using music21's meter awareness.
    
    Args:
        note_obj: The note to check
        time_signature: Optional time signature context
        
    Returns:
        1 if strong downbeat (beat 1), 0 otherwise
    """
    # Primary method: Use music21's beat property (requires measure context)
    try:
        beat = note_obj.beat
        # Beat 1.0 is the strong downbeat in all time signatures
        if beat == 1.0:
            return 1
        return 0
    except:
        pass
    
    # Fallback for MIDI files without time signature / measure context
    # Common in pop songs - assume 4/4 time
    try:
        offset = float(note_obj.offset)
        # In 4/4 time, beat 1 falls every 4 quarter notes
        if offset % 4.0 == 0.0:
            return 1
        return 0
    except:
        return 0


def extract_features(notes: List[note.Note]) -> List[NoteFeature]:
    """
    Extract feature tuples from a list of notes.
    Implements transposition and tempo invariance.
    
    Features:
    - Δp: Pitch interval in semitones (transposition invariant)
    - Δd: Duration ratio (tempo invariant)
    - is_downbeat: Binary flag for metric importance
    
    Args:
        notes: List of music21 Note objects
        
    Returns:
        List of NoteFeature objects
    """
    if len(notes) < 2:
        return []
    
    features = []
    
    for i in range(1, len(notes)):
        prev_note = notes[i - 1]
        curr_note = notes[i]
        
        # Pitch interval (semitones) - transposition invariant
        pitch_interval = curr_note.pitch.midi - prev_note.pitch.midi
        
        # Duration ratio - tempo invariant
        # Use quantized durations (snapped to standard note values)
        prev_duration = prev_note.quarterLength if prev_note.quarterLength > 0 else 0.25
        curr_duration = curr_note.quarterLength if curr_note.quarterLength > 0 else 0.25
        
        # Clamp extreme ratios to reduce noise from ornaments/grace notes
        duration_ratio = curr_duration / prev_duration
        duration_ratio = max(0.125, min(8.0, duration_ratio))
        
        # Downbeat flag
        downbeat = is_downbeat(curr_note)
        
        features.append(NoteFeature(
            pitch_interval=pitch_interval,
            duration_ratio=duration_ratio,
            is_downbeat=downbeat
        ))
    
    return features


def extract_features_from_midi(filepath: str, quantize: bool = True) -> List[NoteFeature]:
    """
    Convenience function to extract features directly from a MIDI file.
    
    Args:
        filepath: Path to the MIDI file
        quantize: Whether to quantize notes to reduce timing noise
        
    Returns:
        List of NoteFeature objects
    """
    notes = parse_midi(filepath, quantize=quantize)
    return extract_features(notes)


def features_to_tuples(features: List[NoteFeature]) -> List[Tuple[float, float, int]]:
    """
    Convert NoteFeature objects to simple tuples for processing.
    
    Args:
        features: List of NoteFeature objects
        
    Returns:
        List of (pitch_interval, duration_ratio, is_downbeat) tuples
    """
    return [f.to_tuple() for f in features]
