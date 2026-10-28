//! The colour encodings this node takes in its parameters.

use camera_core::encoding::ColorEncoding;

/// The encoding `name` spells in any case.
pub(crate) fn parse_encoding(name: &str) -> Result<ColorEncoding, String> {
    ColorEncoding::from_name(&name.to_lowercase()).ok_or_else(|| {
        format!("Invalid encoding '{name}'. Supported encodings are: rgb8, bgr8, mjpeg, yuyv")
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_encoding_is_taken_in_any_case() {
        for encoding in ColorEncoding::ALL {
            assert_eq!(parse_encoding(encoding.name()), Ok(encoding));
            assert_eq!(
                parse_encoding(&encoding.name().to_uppercase()),
                Ok(encoding)
            );
        }
    }

    #[test]
    fn an_unknown_encoding_names_the_supported_ones() {
        assert_eq!(
            parse_encoding("z16"),
            Err(
                "Invalid encoding 'z16'. Supported encodings are: rgb8, bgr8, mjpeg, yuyv"
                    .to_string()
            )
        );
    }
}
