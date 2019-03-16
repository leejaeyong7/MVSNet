class MVSMatch(Match):
    def __init__(self, source_image, target_image, score):
        Match.__init__(source_image, target_image)
        self.score = score

    def to_string(self):
        '{} {} {}'.format(self.source.index, self.target.index, score)
