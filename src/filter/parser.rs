//! Recursive descent parser for SQL-like filter expressions.
//!
//! Grammar:
//!   expr     = or_expr
//!   or_expr  = and_expr ("OR" and_expr)*
//!   and_expr = unary ("AND" unary)*
//!   unary    = "NOT" unary | primary
//!   primary  = "(" expr ")" | comparison
//!   comparison = IDENT "=" STRING
//!              | IDENT "!=" STRING
//!              | IDENT "IN" "(" STRING ("," STRING)* ")"
//!              | IDENT "CONTAINS" STRING

/// Parsed filter expression tree.
#[derive(Debug, Clone, PartialEq)]
pub enum FilterExpr {
    Eq(String, String),
    Ne(String, String),
    In(String, Vec<String>),
    Contains(String, String),
    And(Box<FilterExpr>, Box<FilterExpr>),
    Or(Box<FilterExpr>, Box<FilterExpr>),
    Not(Box<FilterExpr>),
}

/// Parse error.
#[derive(Debug, Clone)]
pub struct ParseError {
    pub message: String,
    pub position: usize,
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "parse error at {}: {}", self.position, self.message)
    }
}

impl std::error::Error for ParseError {}

struct Parser {
    tokens: Vec<Token>,
    pos: usize,
}

#[derive(Debug, Clone, PartialEq)]
enum Token {
    Ident(String),
    StringLit(String),
    Eq,
    Ne,
    LParen,
    RParen,
    Comma,
    And,
    Or,
    Not,
    In,
    Contains,
    Eof,
}

/// Parse a filter expression string into an AST.
pub fn parse_filter(input: &str) -> Result<FilterExpr, ParseError> {
    let tokens = tokenize(input)?;
    let mut parser = Parser { tokens, pos: 0 };
    let expr = parser.parse_or()?;
    if !parser.is_at_end() {
        return Err(ParseError {
            message: format!("unexpected token: {:?}", parser.peek()),
            position: parser.pos,
        });
    }
    Ok(expr)
}

fn tokenize(input: &str) -> Result<Vec<Token>, ParseError> {
    let mut tokens = Vec::new();
    let chars: Vec<char> = input.chars().collect();
    let mut i = 0;

    while i < chars.len() {
        match chars[i] {
            ' ' | '\t' | '\n' | '\r' => i += 1,
            '(' => {
                tokens.push(Token::LParen);
                i += 1;
            }
            ')' => {
                tokens.push(Token::RParen);
                i += 1;
            }
            ',' => {
                tokens.push(Token::Comma);
                i += 1;
            }
            '=' => {
                tokens.push(Token::Eq);
                i += 1;
            }
            '!' if i + 1 < chars.len() && chars[i + 1] == '=' => {
                tokens.push(Token::Ne);
                i += 2;
            }
            '\'' | '"' => {
                let quote = chars[i];
                i += 1;
                let start = i;
                while i < chars.len() && chars[i] != quote {
                    i += 1;
                }
                if i >= chars.len() {
                    return Err(ParseError {
                        message: "unterminated string".to_string(),
                        position: start,
                    });
                }
                let s: String = chars[start..i].iter().collect();
                tokens.push(Token::StringLit(s));
                i += 1;
            }
            c if c.is_alphanumeric() || c == '_' => {
                let start = i;
                while i < chars.len() && (chars[i].is_alphanumeric() || chars[i] == '_') {
                    i += 1;
                }
                let word: String = chars[start..i].iter().collect();
                match word.to_uppercase().as_str() {
                    "AND" => tokens.push(Token::And),
                    "OR" => tokens.push(Token::Or),
                    "NOT" => tokens.push(Token::Not),
                    "IN" => tokens.push(Token::In),
                    "CONTAINS" => tokens.push(Token::Contains),
                    _ => tokens.push(Token::Ident(word)),
                }
            }
            _ => {
                return Err(ParseError {
                    message: format!("unexpected character: '{}'", chars[i]),
                    position: i,
                });
            }
        }
    }

    tokens.push(Token::Eof);
    Ok(tokens)
}

impl Parser {
    fn peek(&self) -> &Token {
        &self.tokens[self.pos]
    }

    fn advance(&mut self) -> &Token {
        let tok = &self.tokens[self.pos];
        if self.pos < self.tokens.len() - 1 {
            self.pos += 1;
        }
        tok
    }

    fn is_at_end(&self) -> bool {
        matches!(self.peek(), Token::Eof)
    }

    fn expect_ident(&mut self) -> Result<String, ParseError> {
        match self.advance().clone() {
            Token::Ident(s) => Ok(s),
            other => Err(ParseError {
                message: format!("expected identifier, got {:?}", other),
                position: self.pos,
            }),
        }
    }

    fn expect_string(&mut self) -> Result<String, ParseError> {
        match self.advance().clone() {
            Token::StringLit(s) => Ok(s),
            other => Err(ParseError {
                message: format!("expected string literal, got {:?}", other),
                position: self.pos,
            }),
        }
    }

    fn expect(&mut self, expected: &Token) -> Result<(), ParseError> {
        let tok = self.advance().clone();
        if &tok == expected {
            Ok(())
        } else {
            Err(ParseError {
                message: format!("expected {:?}, got {:?}", expected, tok),
                position: self.pos,
            })
        }
    }

    fn parse_or(&mut self) -> Result<FilterExpr, ParseError> {
        let mut left = self.parse_and()?;
        while matches!(self.peek(), Token::Or) {
            self.advance();
            let right = self.parse_and()?;
            left = FilterExpr::Or(Box::new(left), Box::new(right));
        }
        Ok(left)
    }

    fn parse_and(&mut self) -> Result<FilterExpr, ParseError> {
        let mut left = self.parse_unary()?;
        while matches!(self.peek(), Token::And) {
            self.advance();
            let right = self.parse_unary()?;
            left = FilterExpr::And(Box::new(left), Box::new(right));
        }
        Ok(left)
    }

    fn parse_unary(&mut self) -> Result<FilterExpr, ParseError> {
        if matches!(self.peek(), Token::Not) {
            self.advance();
            let inner = self.parse_unary()?;
            return Ok(FilterExpr::Not(Box::new(inner)));
        }
        self.parse_primary()
    }

    fn parse_primary(&mut self) -> Result<FilterExpr, ParseError> {
        if matches!(self.peek(), Token::LParen) {
            self.advance();
            let expr = self.parse_or()?;
            self.expect(&Token::RParen)?;
            return Ok(expr);
        }

        let field = self.expect_ident()?;

        match self.peek().clone() {
            Token::Eq => {
                self.advance();
                let value = self.expect_string()?;
                Ok(FilterExpr::Eq(field, value))
            }
            Token::Ne => {
                self.advance();
                let value = self.expect_string()?;
                Ok(FilterExpr::Ne(field, value))
            }
            Token::In => {
                self.advance();
                self.expect(&Token::LParen)?;
                let mut values = vec![self.expect_string()?];
                while matches!(self.peek(), Token::Comma) {
                    self.advance();
                    values.push(self.expect_string()?);
                }
                self.expect(&Token::RParen)?;
                Ok(FilterExpr::In(field, values))
            }
            Token::Contains => {
                self.advance();
                let value = self.expect_string()?;
                Ok(FilterExpr::Contains(field, value))
            }
            other => Err(ParseError {
                message: format!("expected operator after field '{}', got {:?}", field, other),
                position: self.pos,
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_eq() {
        let expr = parse_filter("name = 'hello'").unwrap();
        assert_eq!(expr, FilterExpr::Eq("name".into(), "hello".into()));
    }

    #[test]
    fn test_parse_ne() {
        let expr = parse_filter("status != 'deleted'").unwrap();
        assert_eq!(expr, FilterExpr::Ne("status".into(), "deleted".into()));
    }

    #[test]
    fn test_parse_and() {
        let expr = parse_filter("a = '1' AND b = '2'").unwrap();
        assert_eq!(
            expr,
            FilterExpr::And(
                Box::new(FilterExpr::Eq("a".into(), "1".into())),
                Box::new(FilterExpr::Eq("b".into(), "2".into()))
            )
        );
    }

    #[test]
    fn test_parse_or() {
        let expr = parse_filter("a = '1' OR b = '2'").unwrap();
        assert_eq!(
            expr,
            FilterExpr::Or(
                Box::new(FilterExpr::Eq("a".into(), "1".into())),
                Box::new(FilterExpr::Eq("b".into(), "2".into()))
            )
        );
    }

    #[test]
    fn test_parse_in() {
        let expr = parse_filter("color IN ('red', 'blue')").unwrap();
        assert_eq!(
            expr,
            FilterExpr::In("color".into(), vec!["red".into(), "blue".into()])
        );
    }

    #[test]
    fn test_parse_contains() {
        let expr = parse_filter("tags CONTAINS 'rust'").unwrap();
        assert_eq!(
            expr,
            FilterExpr::Contains("tags".into(), "rust".into())
        );
    }

    #[test]
    fn test_parse_not() {
        let expr = parse_filter("NOT active = 'false'").unwrap();
        assert_eq!(
            expr,
            FilterExpr::Not(Box::new(FilterExpr::Eq("active".into(), "false".into())))
        );
    }

    #[test]
    fn test_parse_parens() {
        let expr = parse_filter("(a = '1' OR b = '2') AND c = '3'").unwrap();
        assert_eq!(
            expr,
            FilterExpr::And(
                Box::new(FilterExpr::Or(
                    Box::new(FilterExpr::Eq("a".into(), "1".into())),
                    Box::new(FilterExpr::Eq("b".into(), "2".into()))
                )),
                Box::new(FilterExpr::Eq("c".into(), "3".into()))
            )
        );
    }

    #[test]
    fn test_parse_double_quotes() {
        let expr = parse_filter("name = \"hello world\"").unwrap();
        assert_eq!(
            expr,
            FilterExpr::Eq("name".into(), "hello world".into())
        );
    }

    #[test]
    fn test_parse_error() {
        assert!(parse_filter("= 'bad'").is_err());
        assert!(parse_filter("field ??").is_err());
        assert!(parse_filter("field = ").is_err());
    }

    #[test]
    fn test_precedence_and_before_or() {
        // a OR b AND c should parse as a OR (b AND c)
        let expr = parse_filter("a = '1' OR b = '2' AND c = '3'").unwrap();
        assert_eq!(
            expr,
            FilterExpr::Or(
                Box::new(FilterExpr::Eq("a".into(), "1".into())),
                Box::new(FilterExpr::And(
                    Box::new(FilterExpr::Eq("b".into(), "2".into())),
                    Box::new(FilterExpr::Eq("c".into(), "3".into()))
                ))
            )
        );
    }
}
