//! The two streams a D4xx opens, and a value held per stream.

/// One of the two streams the node opens.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stream {
    Color,
    Depth,
}

impl Stream {
    /// The stream's name in a geometry refusal.
    pub const fn label(self) -> &'static str {
        match self {
            Self::Color => "colour",
            Self::Depth => "depth",
        }
    }
}

/// One value per stream.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StreamPair<T> {
    pub color: T,
    pub depth: T,
}

impl StreamPair<Stream> {
    /// Each stream, held by itself.
    pub const STREAMS: Self = Self {
        color: Stream::Color,
        depth: Stream::Depth,
    };
}

impl<T> StreamPair<T> {
    /// The value of `stream`, mutably.
    pub fn get_mut(&mut self, stream: Stream) -> &mut T {
        match stream {
            Stream::Color => &mut self.color,
            Stream::Depth => &mut self.depth,
        }
    }

    /// The pair `f` makes of each value, colour first, or the first error.
    pub fn try_map<U, E>(self, mut f: impl FnMut(T) -> Result<U, E>) -> Result<StreamPair<U>, E> {
        Ok(StreamPair {
            color: f(self.color)?,
            depth: f(self.depth)?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_labels_are_the_spellings_of_the_geometry_refusals() {
        assert_eq!(Stream::Color.label(), "colour");
        assert_eq!(Stream::Depth.label(), "depth");
    }

    #[test]
    fn try_map_runs_colour_first_and_stops_at_the_first_error() {
        let mut seen = Vec::new();
        let result: Result<StreamPair<()>, Stream> = StreamPair::STREAMS.try_map(|stream| {
            seen.push(stream);
            Err(stream)
        });
        assert_eq!(result, Err(Stream::Color));
        assert_eq!(seen, [Stream::Color]);
    }

    #[test]
    fn get_mut_reaches_the_value_of_each_stream() {
        let mut pair = StreamPair { color: 1, depth: 2 };
        *pair.get_mut(Stream::Depth) += 10;
        *pair.get_mut(Stream::Color) += 20;
        assert_eq!(
            pair,
            StreamPair {
                color: 21,
                depth: 12
            }
        );
    }
}
