def post_processing(output):
    output_dist = output.squeeze().div(0.8).exp()
    prob = torch.multinomial(output_dist, 2)
    return prob
