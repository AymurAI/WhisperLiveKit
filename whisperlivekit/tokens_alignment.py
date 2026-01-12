from time import time
from typing import Any, Dict, List, Optional, Tuple, Union

from whisperlivekit.timed_objects import (
    ASRToken,
    Segment,
    PuncSegment,
    Silence,
    SilentSegment,
    SpeakerSegment,
    TimedText,
)


class TokensAlignment:
    def __init__(self, state: Any, args: Any, sep: Optional[str]) -> None:
        self.state = state
        self.diarization = args.diarization
        self._tokens_index: int = 0
        self._diarization_index: int = 0
        self._translation_index: int = 0

        self.all_tokens: List[ASRToken] = []
        self.all_diarization_segments: List[SpeakerSegment] = []
        self.all_translation_segments: List[Any] = []

        self.new_tokens: List[ASRToken] = []
        self.new_diarization: List[SpeakerSegment] = []
        self.new_translation: List[Any] = []
        self.new_translation_buffer: Union[TimedText, str] = TimedText()
        self.new_tokens_buffer: List[Any] = []
        self.sep: str = sep if sep is not None else " "
        self.beg_loop: Optional[float] = None
        self.speaker_ids: Dict[int, str] = {}
        self.speaker_candidates: Dict[str, List[Dict[str, Any]]] = {}
        self.candidates_final: bool = False

        self.validated_segments: List[Segment] = []
        self.current_line_tokens: List[ASRToken] = []
        self.diarization_buffer: List[ASRToken] = []

        self.last_punctuation = None
        self.last_uncompleted_punc_segment: PuncSegment = None
        self.unvalidated_tokens: PuncSegment = []

    def update(self) -> None:
        """Drain state buffers into the running alignment context."""
        self.new_tokens, self.state.new_tokens = self.state.new_tokens, []
        self.new_diarization, self.state.new_diarization = (
            self.state.new_diarization,
            [],
        )
        self.new_translation, self.state.new_translation = (
            self.state.new_translation,
            [],
        )
        self.new_tokens_buffer, self.state.new_tokens_buffer = (
            self.state.new_tokens_buffer,
            [],
        )

        self.all_tokens.extend(self.new_tokens)
        self.all_diarization_segments.extend(self.new_diarization)
        self.all_translation_segments.extend(self.new_translation)
        self.new_translation_buffer = self.state.new_translation_buffer

    def add_translation(self, segment: Segment) -> None:
        """Append translated text segments that overlap with a segment."""
        if segment.translation is None:
            segment.translation = ""
        for ts in self.all_translation_segments:
            if ts.is_within(segment):
                segment.translation += ts.text + (self.sep if ts.text else "")
            elif segment.translation:
                break

    def compute_punctuations_segments(
        self, tokens: Optional[List[ASRToken]] = None
    ) -> List[PuncSegment]:
        """Group tokens into segments split by punctuation and explicit silence."""
        segments = []
        segment_start_idx = 0
        for i, token in enumerate(self.all_tokens):
            if token.is_silence():
                previous_segment = PuncSegment.from_tokens(
                    tokens=self.all_tokens[segment_start_idx:i],
                )
                if previous_segment:
                    segments.append(previous_segment)
                segment = PuncSegment.from_tokens(tokens=[token], is_silence=True)
                segments.append(segment)
                segment_start_idx = i + 1
            else:
                if token.has_punctuation():
                    segment = PuncSegment.from_tokens(
                        tokens=self.all_tokens[segment_start_idx : i + 1],
                    )
                    segments.append(segment)
                    segment_start_idx = i + 1

        final_segment = PuncSegment.from_tokens(
            tokens=self.all_tokens[segment_start_idx:],
        )
        if final_segment:
            segments.append(final_segment)
        return segments

    def compute_new_punctuations_segments(self) -> List[PuncSegment]:
        new_punc_segments = []
        segment_start_idx = 0
        self.unvalidated_tokens += self.new_tokens
        for i, token in enumerate(self.unvalidated_tokens):
            if token.is_silence():
                previous_segment = PuncSegment.from_tokens(
                    tokens=self.unvalidated_tokens[segment_start_idx:i],
                )
                if previous_segment:
                    new_punc_segments.append(previous_segment)
                segment = PuncSegment.from_tokens(tokens=[token], is_silence=True)
                new_punc_segments.append(segment)
                segment_start_idx = i + 1
            else:
                if token.has_punctuation():
                    segment = PuncSegment.from_tokens(
                        tokens=self.unvalidated_tokens[segment_start_idx : i + 1],
                    )
                    new_punc_segments.append(segment)
                    segment_start_idx = i + 1

        self.unvalidated_tokens = self.unvalidated_tokens[segment_start_idx:]
        return new_punc_segments

    def concatenate_diar_segments(self) -> List[SpeakerSegment]:
        """Merge consecutive diarization slices that share the same speaker."""
        if not self.all_diarization_segments:
            return []
        merged = [self.all_diarization_segments[0]]
        for segment in self.all_diarization_segments[1:]:
            if segment.speaker == merged[-1].speaker:
                merged[-1].end = segment.end
            else:
                merged.append(segment)
        return merged

    @staticmethod
    def intersection_duration(seg1: TimedText, seg2: TimedText) -> float:
        """Return the overlap duration between two timed segments."""
        start = max(seg1.start, seg2.start)
        end = min(seg1.end, seg2.end)

        return max(0, end - start)

    def get_lines_diarization(self) -> Tuple[List[Segment], str]:
        """Build segments when diarization is enabled and track overflow buffer."""
        diarization_buffer = ""
        punctuation_segments = self.compute_punctuations_segments()
        diarization_segments = self.concatenate_diar_segments()
        for punctuation_segment in punctuation_segments:
            if not punctuation_segment.is_silence():
                if (
                    diarization_segments
                    and punctuation_segment.start >= diarization_segments[-1].end
                ):
                    diarization_buffer += punctuation_segment.text
                else:
                    max_overlap = 0.0
                    max_overlap_speaker = 1
                    for diarization_segment in diarization_segments:
                        intersec = self.intersection_duration(
                            punctuation_segment, diarization_segment
                        )
                        if intersec > max_overlap:
                            max_overlap = intersec
                            max_overlap_speaker = diarization_segment.speaker + 1
                    punctuation_segment.speaker = max_overlap_speaker
                    speaker_id = self.speaker_ids.get(max_overlap_speaker - 1)
                    punctuation_segment.speaker_id = speaker_id
                    if speaker_id and self.speaker_candidates:
                        punctuation_segment.speaker_candidates = (
                            self.speaker_candidates.get(speaker_id)
                        )
                        if punctuation_segment.speaker_candidates is not None:
                            punctuation_segment.candidates_final = (
                                self.candidates_final
                            )

        segments = []
        if punctuation_segments:
            segments = [punctuation_segments[0]]
            for segment in punctuation_segments[1:]:
                if segment.speaker == segments[-1].speaker:
                    if segments[-1].text:
                        segments[-1].text += segment.text
                    segments[-1].end = segment.end
                    if not segments[-1].speaker_id:
                        segments[-1].speaker_id = segment.speaker_id
                    if not segments[-1].speaker_candidates:
                        segments[-1].speaker_candidates = segment.speaker_candidates
                        segments[-1].candidates_final = segment.candidates_final
                else:
                    segments.append(segment)

        # Mark finalities: with diarization, segments split on punctuation/silence
        # are considered final except possibly the last (growing) one.
        for i, seg in enumerate(segments):
            seg.is_final = i < len(segments) - 1

        return segments, diarization_buffer

    def get_lines(
        self,
        diarization: bool = False,
        translation: bool = False,
        current_silence: Optional[Silence] = None,
    ) -> Tuple[List[Segment], str, Union[str, TimedText]]:
        """Return the formatted segments plus buffers, optionally with diarization/translation."""
        if diarization:
            segments, diarization_buffer = self.get_lines_diarization()
        else:
            diarization_buffer = ""
            for token in self.new_tokens:
                if token.is_silence():
                    if self.current_line_tokens:
                        self.validated_segments.append(
                            Segment().from_tokens(self.current_line_tokens)
                        )
                        self.current_line_tokens = []

                    end_silence = (
                        token.end if token.has_ended else time() - self.beg_loop
                    )
                    if (
                        self.validated_segments
                        and self.validated_segments[-1].is_silence()
                    ):
                        self.validated_segments[-1].end = end_silence
                    else:
                        self.validated_segments.append(
                            SilentSegment(start=token.start, end=end_silence)
                        )
                else:
                    self.current_line_tokens.append(token)

            segments = list(self.validated_segments)
            if self.current_line_tokens:
                segments.append(Segment().from_tokens(self.current_line_tokens))

        # Mark finalities: validated segments are final; the last in-progress is not.
        for i, seg in enumerate(segments):
            seg.is_final = True
        if segments:
            segments[-1].is_final = False

        if current_silence:
            end_silence = (
                current_silence.end
                if current_silence.has_ended
                else time() - self.beg_loop
            )
            if segments and segments[-1].is_silence():
                segments[-1] = SilentSegment(start=segments[-1].start, end=end_silence)
            else:
                segments.append(
                    SilentSegment(start=current_silence.start, end=end_silence)
                )
        if translation:
            [
                self.add_translation(segment)
                for segment in segments
                if not segment.is_silence()
            ]
        return segments, diarization_buffer, self.new_translation_buffer.text
